# ingreadient-check
application to read ingredient labels on items of varying shape.
The application compares the ingredients found, and compares it
to the dietary restrictions the user has.

 TODO: adjust/make adaptable to other videos.
 TODO: make frame/image edit class more consistent/easy to use.
 TODO: Update frame merging steps to better manage frame merging failures.

 TODO: decide when to enhance text/ find best way to make label readable.
 TODO: eventually remove draw lines in text direction method
 TODO: test to crop as large as possible rectangular image from by-text-rotated image
 TODO: Stretch/warp image from lines and angles in rotate_image_by_line
       (blur, lower criteria for detection, correct curved
       result to straight lines?)
 TODO: some images look good, except seeming to have higher
       transparency, make all of the image fully opaque(adaptive threshold?)
       and remove shine
TODO: Test label_video_reader with more videos and images to modify and make application
      usable for all kinds of ingredients lists.
TODO: Make all code more uniform, clear and easy to read.
TODO: Figure out how to get a good rectangle/trapezoid from roi detection.
TODO: Try Oriented FAST and Rotated Brief.
TODO: Attempt to crop out the least curved part of label and save for final merge.
TODO: Try to add control of distance between images in each set of frames to merge.