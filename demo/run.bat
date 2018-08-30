exe\deep_image_analogy.exe models\deep_image_analogy\ example\ 0 2 0

exe\similarity_combo.exe models\similarity_subnet\ example\ 0 2 0

python ..\colorization_subnet\test.py --short_size 256 --test_model models\colorization_subnet\example_net.pth --data_root example\ --out_dir example\res\ --gpu_id 0

pause
