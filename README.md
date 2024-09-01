# ingreadient-check
application to read ingredient labels on items of varying shape.
The application compares the ingredients found, and compares it
to the dietary restrictions the user has.

1) record labels to make application usable on containers with label wrapped around.

2) add option to add label by photo for fully visible labels.

3) Detect words on ingredient label, filter and save by target language.

4) Databases & relations, make option to connect detected ingredient list to specific product(and barcode?/option to scan just by barcode to avoid DINO/SAM when not necessary?)

5) GUI, user accounts

TODO:
 1-Better selection of what images to include in panorama
   Explore options such as/similar to object tracking. 
 2-Tune parameters for stitching better-make stitching happen,
   make stitching happen while filming, not after.
 3-Explore using pytesseract readability as guidelines for stitcher and flattening.
 4-Remove unused code, fix dockstings.
 5-check what libraries to keep or remove.
 6-increase clarity for future GUI and database integration.


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
