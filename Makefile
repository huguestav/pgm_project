data:
	tar -zxf Corel_Dataset/labels_raw.tar.gz -C Corel_Dataset/
	rm Corel_Dataset/labels_raw.tar.gz
	tar -zxf Sowerby_Dataset/labels_raw.tar.gz -C Sowerby_Dataset/
	rm Sowerby_Dataset/labels_raw.tar.gz
	tar -zxf Corel_Dataset/images_rgb.tar.gz -C Corel_Dataset/
	rm Corel_Dataset/images_rgb.tar.gz
	tar -zxf Sowerby_Dataset/images_rgb.tar.gz -C Sowerby_Dataset/
	rm Sowerby_Dataset/images_rgb.tar.gz
	python make_datasets.py
filter:
	python doc/color_feature.py

