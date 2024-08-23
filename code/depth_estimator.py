"""
File for test if depth estimation could help with
correcting the frames.

diff of max & min in label area = save, to not run this all the time
run this once or a few times? when?
TODO: make this take cylinders/circles into consideration
"""
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error


class DepthCorrection:
    """
    Class that finds depth and
    correct the image's perspective/flatten it.
    init sets min/max diff
    method just correct by values?
    """
    def __init__(self, frame: np.ndarray) -> None:
        masked_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        self.correct_image(frame=frame_bgra, masked=masked_img)

    def correct_image(self, frame: np.ndarray,
                      masked: np.ndarray) -> np.ndarray:
        """
        use estimated size to correct
        the images shape and perspective
        frame: BGRA image
        depth_mask: GRAYSCALE image
        """
        map_a, map_b = self.get_flattening_maps(masked)
        cv2.imwrite('map_b.png', map_b)
        cv2.imwrite('map_a.png', map_a)
        flattened_image = cv2.remap(frame, map_a, map_b,
                                    interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_WRAP)

        gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2GRAY)
        flattened_image = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2BGR)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        flattened_image = self.inpaint_img(flattened_image, mask)
        cv2.imwrite('flat_img.png', flattened_image)
        self.frame = flattened_image

    def get_flattening_maps(self, masked):
        """
        Method to create 2 maps for flattening/remapping.
        TODO:use difference between min and max
            and use the difference to estimate distance.
        TODO:(?) update this method(or code in general) to create the 2 maps
            at the same time. differences  in min/max distance to allow pixel
            distribution that consider depth when flattening.
        TODO: Update with separate method to call for each map making.
        TODO: Check the detection of edges compared to upside down img.
        TODO: make sure edge detection for top and bottom are done correct.
              double check what is happening.
        """
        map_base_a = masked
        map_base_b = cv2.rotate(masked, cv2.ROTATE_90_CLOCKWISE)

        edge_points_a = self.get_edge_points(map_base_a, True)
        edge_points_b = self.get_edge_points(map_base_b, True)

        pixels_a_start, pixels_a_end, pixels_b_start, pixels_b_end \
            = self.distribute_by_shape(edge_points_a, edge_points_b)

        masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        masked = cv2.flip(masked, -1)
        # Seems to, in most cases fit edges better when flipped upside down?.
        # test with another video seems to confirm this.
        pixel_map_a, masked = self.get_maps(True,
                                            pixels_a_start, pixels_a_end,
                                            pixels_b_start, pixels_b_end,
                                            len(map_base_a[0]), masked,
                                            (255, 255, 0), (0, 0, 255))
        pixel_map_b, masked = self.get_maps(False,
                                            pixels_b_start, pixels_b_end,
                                            pixels_a_start, pixels_a_end,
                                            len(map_base_b[0]), masked,
                                            (255, 0, 255), (0, 255, 0))
        masked = cv2.flip(masked, -1)
        cv2.imwrite('points.png', masked)

        # Transform maps for remapping and return them.
        pixel_map_a = np.array(pixel_map_a).astype(np.float32)
        pixel_map_b = cv2.flip(cv2.rotate(np.array(pixel_map_b),
                               cv2.ROTATE_90_COUNTERCLOCKWISE
                                          ), 0).astype(np.float32)

        return pixel_map_a, pixel_map_b

    def get_edge_points(self, map_base: np.ndarray,
                        reverse: bool = False) -> list:
        """
        indexes of where ROI of each row of the map start and end.
        Return as list of pairs for each row.
        """
        edge_points = []
        for pixel_row in map_base:
            roi = [idx_nr for idx_nr, pix in enumerate(pixel_row) if pix > 0]
            if len(roi) < 1:
                edge_points.append(edge_points[-1])
            else:
                edge_points.append((max(roi), min(roi)))
        if reverse:
            edge_points = [pair[::-1] for pair in edge_points]
        return edge_points

    def distribute_by_shape(self, edge_points_a, edge_points_b):
        """
        estimate the four sides of ROI
        """
        # Get lists of min\max indexes,
        # and adjust them to fit to first or second grade equations
        # A
        pixels_a_start = [edge_point[0] for edge_point in edge_points_a]
        pixels_a_end = [edge_point[1] for edge_point in edge_points_a]
        # B
        pixels_b_start = [edge_point[0] for edge_point in edge_points_b]
        pixels_b_end = [edge_point[1] for edge_point in edge_points_b]

        pixels_a_start, pixels_a_end, pixels_b_start, pixels_b_end\
            = self.normalize_values(pixels_a_start, pixels_a_end,
                                    pixels_b_start, pixels_b_end)

        return pixels_a_start, pixels_a_end, pixels_b_start, pixels_b_end

    def normalize_values(self,
                         pixels_a_start, pixels_a_end,
                         pixels_b_start, pixels_b_end) -> list:
        to_best_fit = []
        for values in [pixels_a_start, pixels_a_end,
                       pixels_b_start, pixels_b_end]:
            values = np.array(values)
            median = np.median(values)
            std = np.std(values)
            values[values < median-std*2] = median-std
            values[values > median+std*2] = median+std
            to_best_fit.append(values)

        pixels_a_start, pixels_a_end, pixels_b_start, pixels_b_end\
            = self.choose_best_fit(to_best_fit)
        return pixels_a_start, pixels_a_end, pixels_b_start, pixels_b_end

    def fit_to_line(self, y):
        x = np.arange(len(y)).reshape(-1, 1)
        y = np.array(y)

        # model = LinearRegression() # decent
        # model = LogisticRegression() # not very good
        model = Ridge()
        model.fit(x, y)

        y_fit = model.predict(x)
        return y_fit

    def fit_to_quadratic(self, y, alpha=0.5, l1_ratio=0.7):
        x = np.arange(len(y)).reshape(-1, 1)
        y = np.array(y)

        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x)

        # model = LinearRegression()
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(x_poly, y)

        y_fit = model.predict(x_poly)
        return y_fit

    def choose_best_fit(self, pixel_values):
        """
        pixel_values:
        list of 4 arrays,
        idx 0 and 2 are opposite
        idx 1 and 3 are opposite
        This method expects to fit to 2 second grade equations and 2 first.
        TODO: Figure out how to make tis more adaptable to e.g. perspectives
        TODO: explore possibilities of making regression methods more dependent
              on each other(not be to different)
        TODO: Maybe put this in separate class with model for each side control
              outlier problems.
        """
        line_fits = []
        quad_fits = []
        for values in pixel_values:
            line_fits.append(self.fit_to_line(values))
            quad_fits.append(self.fit_to_quadratic(values))

        rmse_line_0_1 = \
            np.sqrt(mean_squared_error(pixel_values[0], line_fits[0])) +\
            np.sqrt(mean_squared_error(pixel_values[1], line_fits[1]))
        rmse_line_2_3 = \
            np.sqrt(mean_squared_error(pixel_values[2], line_fits[2])) +\
            np.sqrt(mean_squared_error(pixel_values[3], line_fits[3]))

        if rmse_line_0_1 > rmse_line_2_3:
            a, b, c, d = quad_fits[0].tolist(), quad_fits[1].tolist(), \
                line_fits[2].tolist(), line_fits[3].tolist()
        else:
            a, b, c, d = line_fits[0].tolist(), line_fits[1].tolist(), \
                quad_fits[2].tolist(), quad_fits[3].tolist()
        return a, b, c, d

    def get_maps(self, first_value: bool,
                 pixels_start: list, pixels_end: list,
                 connected_start: list, connected_end: list,
                 len_active: int, image: np.ndarray,
                 color_1: tuple, color_2: tuple):
        """
        generate the actual maps.
        TODO: make this take width/difference between numbers into account.
              for roi length index at the opposite direction.
        """
        pixel_map = []
        for row_idx, (start, stop) in enumerate(zip(pixels_start,
                                                    pixels_end)):
            pixel_map.append(np.linspace(start, stop, len_active))
            if first_value:
                location_start = (int(start), row_idx)
                location_stop = (int(stop), row_idx)
            else:
                location_start = (row_idx, int(start))
                location_stop = (row_idx, int(stop))

            # LEN(ACTIVE)=indexes of where to find min/max distance/depth.
            # Connected to short for this?/need to make sure it is long enough
            # send in list of lengths instead?
            masked = cv2.circle(image, location_start,
                                1, color_1, 1)
            masked = cv2.circle(masked, location_stop,
                                1, color_2, 1)
        return pixel_map, masked

    def inpaint_img(self, img, mask):
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return img


class GetModel:
    """
    Temporary(?) class to test different
    edge detection regressions
    TODO: Explore CNN?
    """

    def __init__(self, indexes_start, indexes_end, image: np.ndarray):
        """
        Create data frame and find values for ROI regression model.
        input(indexes_start, indexes_end) must be of equal length.
        image height must be of equal length as indexes.
        image must be rotated so that each row includes a start and stop value.
        """
        self.indexes_start, self.indexes_end = indexes_start, indexes_end
        self.roi_data = image
        self.df = self.create_data_frame()

        self.X_train_start, self.X_train_end, \
            self.X_test_start, self.X_test_end, \
            self.Y_train_start, self.Y_train_end, \
            self.Y_test_start, self.Y_test_end = \
            self.split_train_test(['values_start', 'values_end'])

        self.model_start_stats = []
        self.model_end_stats = []

        self.linear_model_start, \
            self.linear_model_end = self.create_linear()
        self.lasso_model_start, \
            self.lasso_model_end = self.create_lasso()
        self.ridge_model_start, \
            self.ridge_model_end = self.create_ridge()
        self.elastic_model_start, \
            self.elastic_model_end = self.create_elastic()
        self.svr_model_start, \
            self.svr_model_end = self.create_svr()

        self.final_model_start, \
            self.final_model_end = self.choose_best_model()

    def create_data_frame(self):
        """
        Convert the np.ndarray to pandas data frame
        fields: index, height/width value, length of ROI(to add later on)
        """
        df = pd.DataFrame({
            'idx': range(len(self.data)),
            'roi_len': [self.find_ROI_length()],
            'values_start': self.indexes_start,
            'values_end': self.indexes_end
        })
        return df

    def find_ROI_length(self) -> list:
        """
        for each start or end point, find number ROI pixels
        in same row or column. Add info to data frame.
        (find in columns if going left to right, find in rows else)
        """
        return np.count_nonzero(self.roi_data, axis=1)

    def split_train_test(self, predict_columns: list):
        """
        Return train and test sets for dependent and target.
        """
        X_train_start, X_test_start, y_train_start, y_test_start = \
            train_test_split(self.df.drop(predict_columns[0]),
                             self.df[predict_columns[0]],
                             test_size=0.2, random_state=101)
        X_train_end, X_test_end, y_train_end, y_test_end = \
            train_test_split(self.df.drop(predict_columns[1]),
                             self.df[predict_columns[1]],
                             test_size=0.2, random_state=101)

        return X_train_start, X_train_end, X_test_start, X_test_end, \
            y_train_start, y_train_end, y_test_start, y_test_end

    def create_linear(self):
        """
        Create evaluation with linear regression
        """
        pipeline_linear = Pipeline(steps=[
            ('scale', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('regression', LinearRegression())])
        param_grid = {'poly_degree': [1, 2, 3, 4]}  # Remove 3 and 4 later?

        linear_pipe_grid = GridSearchCV(pipeline_linear, param_grid,
                                        cv=10, scoring='r2')
        linear_start = linear_pipe_grid.fit(self.X_train_start,
                                            self.Y_train_start)
        linear_end = linear_pipe_grid.fit(self.X_train_end,
                                          self.Y_train_end)

        best_start = \
            linear_start.best_estimator_.fit(self.df.drop('values_start'),
                                             self.df['values_start'])
        best_end = \
            linear_end.best_estimator_.fit(self.df.drop('values_end'),
                                           self.df['values_end'])

        return best_start, best_end

    def create_lasso(self):
        """
        Create evaluation with lasso regression
        """
        pipeline_lasso = Pipeline(steps=[
            ('scale', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('regression', Lasso())])
        param_grid = {'poly__degree': [1, 2, 3, 4],
                      'regression__alpha': [0.10, 0.20, 0.30, 0.40, 0.50,
                                            0.60, 0.70, 0.80, 0.90,
                                            1, 5, 10, 50, 75, 100],
                      'regression__tol': [0.3]}

        lasso_pipe_grid = GridSearchCV(pipeline_lasso, param_grid,
                                       cv=10, scoring='r2')
        lasso_start = lasso_pipe_grid.fit(self.X_train_start,
                                          self.Y_train_start)
        lasso_end = lasso_pipe_grid.fit(self.X_train_end,
                                        self.Y_train_end)

        best_start = \
            lasso_start.best_estimator_.fit(self.df.drop('values_start'),
                                            self.df['values_start'])
        best_end = \
            lasso_end.best_estimator_.fit(self.df.drop('values_end'),
                                          self.df['values_end'])

        return best_start, best_end

    def create_ridge(self):
        """
        Create evaluation with ridge model
        """
        pipeline_ridge = Pipeline(steps=[
            ('scale', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('regression', Ridge())])
        param_grid = {'poly__degree': [1, 2, 3, 4],
                      'regression__alpha': [0.1, 0.5, 1, 5, 10, 50, 100],
                      'regression__tol': [0.5]}
        ridge_pipe_grid = GridSearchCV(pipeline_ridge, param_grid,
                                       cv=10, scoring='r2')

        ridge_start = ridge_pipe_grid.fit(self.X_train_start,
                                          self.Y_train_start)
        ridge_end = ridge_pipe_grid.fit(self.X_train_end,
                                        self.Y_train_end)

        best_start = \
            ridge_start.best_estimator_.fit(self.df.drop('values_start'),
                                            self.df['values_start'])
        best_end = \
            ridge_end.best_estimator_.fit(self.df.drop('values_end'),
                                          self.df['values_end'])

        return best_start, best_end

    def create_elastic(self):
        """
        Create evaluation with elastic net model.
        """
        pipeline_elastic = Pipeline(steps=[
            ('scale', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('regression', ElasticNet())])
        param_grid = {'poly__degree': [1, 2, 3, 4],
                      'regression__alpha': [0.1, 0.5, 1, 5, 10, 50, 100],
                      'regression__l1_ratio': [.05, .1, .15, .2, .3, .5,
                                               .7, .9, 95, 99, 1],
                      'regression__tol': [0.5]}
        elastic_pipe_grid = GridSearchCV(pipeline_elastic, param_grid,
                                         cv=10, scoring='r2')

        elastic_start = elastic_pipe_grid.fit(self.X_train_start,
                                              self.Y_train_start)
        elastic_end = elastic_pipe_grid.fit(self.X_train_end,
                                            self.Y_train_end)

        best_start = \
            elastic_start.best_estimator_.fit(self.df.drop('values_start'),
                                              self.df['values_start'])
        best_end = \
            elastic_end.best_estimator_.fit(self.df.drop('values_end'),
                                            self.df['values_end'])

        return best_start, best_end

    def create_svr(self):
        """
        Create evaluation with SVR model
        """
        pipeline_svr = Pipeline(steps=[
            ('scale', StandardScaler()),
            ('regression', SVR())])
        param_grid = {'regression__C': [0.001, 0.01, 0.1, 0.5, 5, 10, 100],
                      'regression__kernel': ['linear', 'poly', 'rbf'],
                      'regression__degree': [2, 3, 4, 5, 6],
                      'regression__epsilon': [0, 0.01, 0.1, 0.5, 1, 2],
                      'regression__gamma': ['scale', 'auto']}
        svr_pipe_grid = GridSearchCV(pipeline_svr, param_grid,
                                     cv=10, scoring='r2')

        svr_start = svr_pipe_grid.fit(self.X_train_start,
                                      self.Y_train_start)
        svr_end = svr_pipe_grid.fit(self.X_train_end,
                                    self.Y_train_end)

        best_start = \
            svr_start.best_estimator_.fit(self.df.drop('values_start'),
                                          self.df['values_start'])
        best_end = \
            svr_end.best_estimator_.fit(self.df.drop('values_end'),
                                        self.df['values_end'])

        return best_start, best_end

    def choose_best_model(self):
        """
        Compare performance of all models and choose the best.
        """
        pass
