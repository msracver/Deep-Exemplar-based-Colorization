mkdir -p /src/app/custom-examples/res/
python3 ../colorization_subnet/test.py --short_size 256 --test_model models/colorization_subnet/example_net.pth --data_root /src/app/custom-examples/ --out_dir /src/app/custom-examples/res/ --gpu_id 0
