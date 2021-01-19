python stitch_patches_pdf.py \
--data_file ../train_pdf \
--output_dir ../eval/stitched_real_world_iter1/ \
--math_dir ../eval/test_real_world_iter1/ \
--stitching_algo equal  \
--algo_threshold 30 \
--num_workers 1 \
--postprocess True \
--home_images ../images/