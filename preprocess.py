import pathlib
import numpy as np
import torchvision
from PIL import Image, ImageOps
import argparse
import time

def preprocess(source_dir, dest_dir, img_size):
    t1 = time.time()
    segmentation_image_paths = list(pathlib.Path.glob(source_dir, "*_Segmentation.*"))
    image_paths = list(pathlib.Path.glob(source_dir, "*[!Segmentation].*"))
    pathlib.Path.mkdir(dest_dir, parents=True, exist_ok=True)
    

    for image, seg in zip(image_paths, segmentation_image_paths):
        seg_im = Image.open(seg)
        img = Image.open(image)
        seg_im = ImageOps.exif_transpose(seg_im)
        img = ImageOps.exif_transpose(img)
        seg_im= np.array(seg_im).astype(dtype=np.float32)
        seg_im = seg_im / 255
        
        seg_im_sum_ax0 = np.squeeze(np.sum(seg_im, axis=0))
        L = -1; R = -1
        for i, col in enumerate(seg_im_sum_ax0):
            if col !=0:
                R = i
                if L == -1:
                    L = i
        try:
            assert L >=0 and R >=0
        except:
            raise "Invalid image error"

        seg_im_sum_ax1 = np.squeeze(np.sum(seg_im, axis=1))
        U = -1; B = -1
        for i, row in enumerate(seg_im_sum_ax1):
            if row !=0:
                B = i
                if U == -1:
                    U = i
        try:
            assert U >=0 and B >=0
        except:
            raise "Invalid image error"
        
        img = torchvision.transforms.functional.crop(img, U, L, (B-U), (R-L))
        resize = torchvision.transforms.Resize(img_size)
        img = resize(img)
       
        image_name = image.name
        fp = pathlib.Path(dest_dir / image_name)
        img.save(fp)

        del seg_im
        del img
    
    t2 = time.time()
    
    del segmentation_image_paths
    del image_paths

    return (t2-t1)

def get_args():
    parser = argparse.ArgumentParser(description="Generate preprocessed dataset of cropped images of lesions")
    parser.add_argument("--source_dir", "-s", type=str, default="./ISIC2016_WithMasks_Train", help="Source directory containing original images to be cropped."
                        "\nBoth images and masks."
                        "\nMasks name should be Image name with suffix _Segmentation"
                        )
    parser.add_argument("--dest_dir", "-d", type=str, default="./ISIC2016_TrainingImages",
                        help="Path of dest dir to store cropped images."
                        "Image name same as original"
                        )
    parser.add_argument("--img_size", "-i", nargs=2 , type=int, default=150,
                        help="The size for the cropped images to be resized to and stored."
                        "Give the height and width seperated by a space"
                        )
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_args()

    source_dir = pathlib.Path.cwd() / "ISIC2016_WithMasks_Train"
    dest_dir = pathlib.Path.cwd() / "ISIC2016_TrainingImages"
    
    if args.source_dir != "./ISIC2016_WithMasks_Train":
        source_dir = pathlib.Path(args.source_dir)
    if args.dest_dir != "./ISIC2016_TrainingImages":
        dest_dir = pathlib.Path(args.dest_dir)
    if args.img_size != 150:
        img_size = tuple(args.img_size)
    else:
        img_size = (150, 150)

    time_taken = preprocess(source_dir=source_dir, dest_dir=dest_dir, img_size=img_size)
    
    print(f"Images from {source_dir}"
          f"\nPreprocessed and stored at {dest_dir}"
          f"\nWith img_size={img_size}"
          f"\nIn {time_taken} seconds"
          )
    
    