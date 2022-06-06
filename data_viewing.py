import pydicom as dicom
import matplotlib.pylab as plt

# specify your image path
image_path = 'data/train_images/sample.dcm'
ds = dicom.dcmread(image_path)

plt.imshow(ds.pixel_array)