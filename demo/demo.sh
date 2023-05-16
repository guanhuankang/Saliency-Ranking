rm -rf output/*.png
rm -rf output/*.jpg
python demo.py --config-file ../configs/swinb_srnet.yaml --input examples/*.jpg --output output --opts MODEL.WEIGHTS $1

