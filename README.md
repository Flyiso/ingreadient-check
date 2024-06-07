# ingreadient-check
application to read ingredient labels on items of varying shape.
The application compares the ingredients found, and compares it
to the dietary restrictions the user has.

1) record labels to make application usable on containers with label wrapped around.
TODO: test to crop as large as possible rectangular image from by-text-rotated image
 
TODO: Test label_video_reader with more videos and images to modify and make application
      usable for all kinds of ingredients lists.

TODO: control of distance between images in each merge?.
TODO: Try to use panorama mode for stitcher every x
      merge(with high thresh?) to avoid loosing first img section

TODO: Instead of trying the next image when merge fails, try to
      merge add and instead merge more than 2 frames at a time
      (and put limit to max n of frames in each merge?)
      -seems to work better.
      TODO?: first sections of merged result disappear later on.
             use mask to avoid? or solve in any other way?
TODO: try panorama mode when scans not working?
TODO: try to also warp img before merge- 
      take both images into consideration and adjust accordingly?

TODO: make it runable on mobile devices.
      (less pytesseract? more efficient ways to find text region? image segmentation? huggingface?)

TODO: enhance images before merge?
  DONE: lightness.
  TODO?: blur? bilateral?
  TODO?: enhance/increase contrast of letters
TODO: get mask to cover possible shiny areas of image before merge?

TODO: error when no contour found- create early escape/error management(check DINO)

TODO: Manage error in stitcher, try to add more images.

TODO: ad text read method for when full image is finally merged.

TODO: Instead of calling dino model for each segmentation, re-use first segmented outline & try
      find similar shape in frame? (makes application faster/more suitable for small devices?)
      -TRY -object-tracking- add frame to merge evenly distanced.

TODO: modify for live usage.
      1-segment label- detect label before asking user to keep item at same distance
      and start to turn it. (paint label-shaped box on screen to guide user?)
            a. use that box to detect difference between current and last frame to decide
               what frames to merge? - could make segmentation part of program less demanding?
      2. -Live updates of panorama img- force end for video,(improve quality/merge again afterwards?)
      3. Make it return the final image- connect it all to class/code that connect it to everything else
         needed for database creation.

2) add option to add label by photo for fully visible labels.

3) Detect words on ingredient label, filter and save by target language.

4) Databases & relations, make option to connect detected ingredient list to specific product(and barcode?/option to scan just by barcode to avoid DINO/SAM when not necessary?)

5) GUI, user accounts

-- possible attributes(or relations) for each product:
   id-
   barcode-(same as id?)
   img of ingredients(?)-
   product name
   product company/creator
   timestamp when last update
   ingredients-relational/ingredients might need connections to allergens?
   possibly connections to user notes about product.

## Citation
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
