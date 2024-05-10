# ingreadient-check
application to read ingredient labels on items of varying shape.
The application compares the ingredients found, and compares it
to the dietary restrictions the user has.

1) record labels to make application usable on containers with label wrapped around.
TODO: test to crop as large as possible rectangular image from by-text-rotated image
 
TODO: Test label_video_reader with more videos and images to modify and make application
      usable for all kinds of ingredients lists.

TODO: control of distance between images in each merge?.

TODO: Instead of trying the next image when merge fails, try to
      merge add and instead merge more than 2 frames at a time
      (and put limit to max n of frames in each merge?)
TODO: try panorama mode when scans not working?

TODO: explore how/if to customize stitcher.


TODO: make it runable on mobile devices.
      (less pytesseract? more efficient ways to find text region? image segmentation? huggingface?)

TODO: enhance images before merge?
  DONE: lightness.
  TODO?: blur? bilateral?
  TODO?: enhance/increase contrast of letters
TODO: get mask to cover possible shiny areas of image before merge?

TODO: error when no contour found- create early escape/error management(check DINO)

TODO: Manage error in stitcher, try to add more images.

2) add option to add label by photo for fully visible labels.

3) Detect words on ingredient label, filter and save by target language.

4) Databases & relations, make option to connect detected ingredient list to specific product(and barcode?/option to scan just by barcode to avoid DINO/SAM when not necessary?)

5) GUI, user accounts

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
