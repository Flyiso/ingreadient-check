# ingreadient-check
application to read ingredient labels on items of varying shape.
The application compares the ingredients found, and compares it
to the dietary restrictions the user has.

TODO: adjust/make adaptable to other videos./ new videos added.
TODO: make frame/image edit class more consistent/easy to use. / New class.

TODO: test to crop as large as possible rectangular image from by-text-rotated image
 
TODO: Test label_video_reader with more videos and images to modify and make application
      usable for all kinds of ingredients lists.

TODO: control of distance between images in each merge?.

TODO: explore how/if to customize stitcher.

TODO: create mask and enhancement methods for image management.

TODO: make it runable on mobile devices.
      (less pytesseract? more efficient ways to find text region? image segmentation? huggingface?)

TODO: crop output/detected label from DINO
TODO: Add GroundedSAM model to better capture/correct text perspective?

## Citation
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
