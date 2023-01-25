


# from turtle import colormode
from main_ui import Ui_MainWindow
import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QMessageBox
# from PyQt5.QtCore import QTime,QTimer ,QDate


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,  NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.image 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


import cv2

import pydicom
import numpy as np
from PIL import Image
# import copy


from phantominator import shepp_logan

from back_projection import back_projection, generate_phantom
from frequency_domain_filters import *
from histogram_oprations import *
from interpolation_transilation import *
from misc import *
from morphological_operators import *
from noises import *
from spatial_domain_filters import *



mpl.use('Qt5Agg')




class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4):
        self.fig = Figure()#figsize=(width, height))
        # self.axes = self.fig.add_subplot(111)
        # fig.tight_layout()
        # self.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        
        super(MplCanvas, self).__init__(self.fig)

class grid_MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4):
        self.fig, self.axes = plt.subplots(2,2)
        self.fig.tight_layout()
        
        super(grid_MplCanvas, self).__init__(self.fig)

class row_MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4):
        self.fig, self.axes = plt.subplots(1,2)
        self.fig.tight_layout()
        
        super(row_MplCanvas, self).__init__(self.fig)

class row_3__MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4):
        self.fig, self.axes = plt.subplots(1,3)
        self.fig.tight_layout()
        
        super(row_3__MplCanvas, self).__init__(self.fig)



class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # inserting matplotlib's canvases into the ui
        self.viewer_plot = MplCanvas(self, width=5, height=4)
        self.nearest_neighbor_plot = MplCanvas(self, width=5, height=4)
        self.bilinear_plot = MplCanvas(self, width=5, height=4)
        self.equalization_plot = grid_MplCanvas(self, width=5, height=4)
        self.fourier_transform_plot = grid_MplCanvas(self, width=5, height=4)
        self.unsharp_plot = grid_MplCanvas(self, width=5, height=4)
        self.ROI_plot = row_MplCanvas(self, width=5, height=4)
        self.back_projection_plot = row_3__MplCanvas(self, width=5, height=4)
        self.morphology_plot = row_MplCanvas(self, width=5, height=4)
        
        self.ui.verticalLayout_viewer.addWidget(self.viewer_plot)
        self.ui.verticalLayout_nearst_neighbor.addWidget(self.nearest_neighbor_plot)
        self.ui.verticalLayout_bilinear.addWidget(self.bilinear_plot)
        self.ui.verticalLayout_equalize.addWidget(self.equalization_plot)
        self.ui.verticalLayout_fourier_transform.addWidget(self.fourier_transform_plot)
        self.ui.verticalLayout_unsharp_plot.addWidget(self.unsharp_plot)
        self.ui.verticalLayout_back_projection.addWidget(self.back_projection_plot)
        self.ui.verticalLayout_morphology.addWidget(self.morphology_plot)



        self.ui.label_viewer_info.setText("please open an image")
        self.ui.radioButton_bilinear.setChecked(True)
        

        self.path = ""
        self.img = np.zeros((128,128))
        self.ui.label_interpolation.setText(':::::::::::::::::::)')
        self.bitdepth = 1
        self.morphology_test_kernel = np.array([[0,1,1,1,0],
                                                [1,1,1,1,1],
                                                [1,1,1,1,1],
                                                [1,1,1,1,1],
                                                [0,1,1,1,0]])


        # connecting buttons

        self.ui.actionopen.triggered.connect(self.openimage)
        self.ui.pushButton_apply_interpolation.clicked.connect(self.interpolate)
        self.ui.pushButton_apply_translation.clicked.connect(self.apply_translation)

        self.ui.pushButton_plot_t_shape.clicked.connect(self.set_img_to_t_shape)
        self.ui.pushButton_plot_phantom.clicked.connect(self.set_img_to_phantom)

        self.ui.pushButton_apply_unsharp_filter.clicked.connect(self.apply_unsharp_filters)
        self.ui.pushButton_apply_high_boost_filter.clicked.connect(self.apply_high_boost_filter)

        self.ui.pushButton_add_noise.clicked.connect(self.add_noise)
        self.ui.pushButton_apply_median_filter.clicked.connect(self.apply_median_filter)
        # self.ui.pushButton_remove_noise.clicked.connect(self.remove_noise)

        self.ui.pushButton_apply_selectROI.clicked.connect(self.selectROI)

        self.ui.pushButton_apply_back_projection.clicked.connect(self.apply_back_projection)

        self.ui.pushButton_plot_binary_img.clicked.connect(self.plot_binary_img)
        self.ui.pushButton_apply_erosion.clicked.connect(self.apply_erosion)
        self.ui.pushButton_apply_dilation.clicked.connect(self.apply_dilation)
        self.ui.pushButton_apply_openning.clicked.connect(self.apply_openning)
        self.ui.pushButton_apply_closing.clicked.connect(self.apply_closing)
        self.ui.pushButton_apply_morphological_filter.clicked.connect(self.apply_morphological_filter)

    def openimage(self):
        # try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.path, _ = QFileDialog.getOpenFileName(self,"choose the image", "","images (*jpg *.dcm)")#, options=options)
            if self.path:
                print(self.path)
            if self.path[-3:] == "jpg" :

                self.img = mpl.image.imread(self.path)
                metadata = ''#self.get_metadata('jpg')

                try:
                    self.img = rgb2gray(self.img)
                except:
                    pass
                self.viewer_plot.fig.clf()
                self.viewer_plot.fig.figimage(self.img, cmap='gray')

            elif self.path[-3:] == "dcm":
            
                self.dicom_object = pydicom.dcmread(self.path)

                self.img = self.dicom_object.pixel_array
                

                metadata = self.get_metadata('dcm')


                self.viewer_plot.fig.clf()
                self.viewer_plot.fig.figimage(self.img)

            else:
                metadata = "||| this app support only jpeg and dicom files |||"
            self.ui.label_viewer_info.setText(metadata)
            self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)  
              
        # except:
        #         msg = QMessageBox()
        #         msg.setIcon(QMessageBox.Critical)
        #         msg.setText("the file is corrupted")
        #         msg.setWindowTitle("Error")
        #         msg.exec_()
        #         print("err occurred")
            self.apply_histogram_equalization()
            self.apply_fourier_transform()
    
    def get_metadata(self,imagetype):
        metadata = ''
        if imagetype == 'jpg':
            pillowimg = Image.open(self.path)
            
            numofrows = self.img.shape[0]
            numofcolumns = self.img.shape[1]
            numofbits = numofcolumns * numofrows
            print(pillowimg.getbands)
            self.bitdepth = pillowimg.bits * len(pillowimg.getbands())
            metadata = """ number of rows = {}\n number of coulmns = {}\n number of pixels = {}""".format(numofrows, numofcolumns, numofbits,pillowimg.mode)
            

        elif imagetype == 'dcm':
            
            self.bitdepth =self.dicom_object.BitsAllocated
            metadata = """
            number of rows = {}
            number of coulmns = {}
            number of pixels = {}
            Patient's Name...: {}
            Patient's ID...: {}
            Modality...: {}
            Study Date...: {}
            Pixel Spacing...: {}
            Bit Depth...: {}
            """.format(
                self.dicom_object.Rows,
                self.dicom_object.Columns,
                self.dicom_object.Rows * self.dicom_object.Columns,
                self.dicom_object.PatientName,
                self.dicom_object.PatientID,
                self.dicom_object.Modality,
                self.dicom_object.StudyDate,
                self.dicom_object.PixelSpacing,
                self.dicom_object.BitsAllocated)

        return metadata


        metadata = """
        number of rows = {}
        number of coulmns = {}
        size = {}
        color type = {}
        bitdepth = {}""".format(numofrows, numofcolumns, numofbits * self.bitdepth,pillowimg.mode,self.bitdepth)
        
    def interpolate(self):
        if len(self.img) == 0:
            return
        print(self.img)
        
        zooming_factor = self.ui.doubleSpinBox_zooming_factor.value()
        print(zooming_factor)
        if zooming_factor > 0 :
            self.nn_img = nn_interpolation(self.img,zooming_factor)
            print(self.nn_img)
            # print(self.img)

            self.nearest_neighbor_plot.fig.clf()
            self.nearest_neighbor_plot.fig.figimage(self.nn_img,cmap='gray')

            
            self.bl_img = bl_interpolation(self.img,zooming_factor)
            print(self.bl_img)
            self.bilinear_plot.fig.clf()
            self.bilinear_plot.fig.figimage(self.bl_img,cmap='gray')

            self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32) 


            

            metadata = """
                nearest neighbor:
                number of rows = {} number of coulmns = {} number of pixels = {} -{} color type = {}
                
                """.format(self.nn_img.shape[0], self.nn_img.shape[1], self.nn_img.shape[0]*self.nn_img.shape[1],self.nn_img.shape[0]*self.nn_img.shape[1]*self.bitdepth,'grayscale')

            
            
            metadata = metadata + """
                bilinear:
                number of rows = {}  number of coulmns = {}  number of pixels = {} - {}  color type = {}
                """.format(self.bl_img.shape[0], self.bl_img.shape[1], self.bl_img.shape[0]*self.bl_img.shape[1],self.nn_img.shape[0]*self.nn_img.shape[1]*self.bitdepth,'grayscale')

            self.ui.label_interpolation.setText(metadata)
    
    def apply_translation(self):

        rotatation_angle_degree = self.ui.doubleSpinBox_rotation_angle.value()
        scale_factor = self.ui.doubleSpinBox_scale_factor.value()
        shear_factor = self.ui.doubleSpinBox_shear_factor.value()
        if self.ui.radioButton_nearest_neighbour.isChecked():
            rotation_technique = 'nearest neighbour'
        else:
            rotation_technique = 'bilinear'


        rotatation_angle_rad = rotatation_angle_degree * np.pi/180

        self.img = translate_image(self.img,rotation_angle= rotatation_angle_rad,rotation_technique=rotation_technique,scale_factor=scale_factor,shear_factor=shear_factor)


        self.viewer_plot.fig.clf()
        self.viewer_plot.fig.figimage(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)  
    
    def set_img_to_t_shape(self):
        t_img = generate_t_shape_image()

        self.img = t_img

        self.viewer_plot.fig.clf()
        self.viewer_plot.fig.figimage(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)  

    def apply_histogram_equalization(self):
        eq_img, old_dist, new_dist = histogram_equalization(self.img)

        self.equalization_plot.axes[1,0].clear()
        self.equalization_plot.axes[1,1].clear()

        self.equalization_plot.axes[0,0].imshow(self.img,cmap='gray')
        self.equalization_plot.axes[0,1].imshow(eq_img,cmap='gray')
        self.equalization_plot.axes[1,0].bar(np.arange(0,256),old_dist)
        self.equalization_plot.axes[1,1].bar(np.arange(0,256),new_dist)

        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)  

    def apply_unsharp_filters(self):
        # self.img = rgb2gray(mpl.image.imread(self.path))
        kernel_size = int(self.ui.doubleSpinBox_unsharp_kernal_size.value())
        x = int(np.ceil(kernel_size/3))
        k = self.ui.doubleSpinBox_high_boost_factor.value()
        unsharped_img = spatial_unsharp_box_filter(self.img,kernel_size= kernel_size,k=k)
        self.viewer_plot.fig.figimage(unsharped_img,cmap='gray')

        fft_unsharpped_img = fft_high_boost_filter(self.img,kernel_size= kernel_size,k=k)

        diff_img = scale_pixels_values(  unsharped_img - np.pad(fft_unsharpped_img,x,mode='reflect'))

        self.unsharp_plot.axes[0,0].imshow(scale_pixels_values(unsharped_img),cmap='gray')
        self.unsharp_plot.axes[0,1].imshow(scale_pixels_values(fft_unsharpped_img),cmap='gray')
        self.unsharp_plot.axes[1,0].imshow(diff_img,cmap='gray')



        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)  

    def apply_median_filter(self):
        # self.img = rgb2gray(mpl.image.imread(self.path))
        kernel_size = int(self.ui.doubleSpinBox_median_kernal_size.value())
        median_img = median_filter(self.img,kernel_size= kernel_size)
        self.viewer_plot.fig.figimage(median_img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)  
 
    def add_noise(self):
        noise_type = self.ui.comboBox_noise_type.currentText()
        print(noise_type)
        if noise_type == 'salt and pepper':
            self.img = add_salt_pepper_noise(self.img)
        elif noise_type == 'gaussian':
            self.img = add_gaussian_uniform_noise(self.img,'gaussian')
        elif noise_type == 'uniform':
            self.img = add_gaussian_uniform_noise(self.img,'uniform')

        self.viewer_plot.fig.figimage(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)         

    def apply_fourier_transform(self):
        fft_img = np.fft.fft2(self.img)
        fft_img = np.fft.fftshift(fft_img)
        fft_img_phase = np.angle(fft_img)
        fft_img_mag = np.abs(fft_img)

        

        self.fourier_transform_plot.axes[0,0].imshow(fft_img_mag,cmap='gray')
        self.fourier_transform_plot.axes[0,1].imshow(fft_img_phase,cmap='gray')
        self.fourier_transform_plot.axes[1,0].imshow(np.log(10+fft_img_mag),cmap='gray')
        self.fourier_transform_plot.axes[1,1].imshow(np.log(10+fft_img_phase),cmap='gray')
        print(np.log(1+fft_img_phase))

        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32) 
     
    # def remove_noise(self):
    #     noise_removed_img = remove_periodic_noise(self.img)
    #     # noise_removed_img = notch_reject_filter(self.img.shape)
    #     self.viewer_plot.fig.figimage(noise_removed_img,cmap='gray')
    #     self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)  
    #     self.fourier_transform_plot.axes[1,1].imshow(np.log(np.abs(np.fft.fft2(noise_removed_img))),cmap='gray')


    def set_img_to_phantom(self):
        phantom = generate_phantom()
        self.img = phantom

        self.viewer_plot.fig.clf()
        self.viewer_plot.fig.figimage(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)    
  
    def apply_high_boost_filter(self):

        kernel_size = int(self.ui.doubleSpinBox_unsharp_kernal_size.value())
        k = self.ui.doubleSpinBox_high_boost_factor.value()

        self.img = fft_high_boost_filter(self.img,kernel_size,k)

        self.viewer_plot.fig.clf()
        self.viewer_plot.fig.figimage(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32) 

    def selectROI(self):
        scaled_image = (np.maximum(self.img, 0) / self.img.max()) * 255.0 
        scaled_image = np.uint8(scaled_image)
        roi = cv2.selectROI('select',scaled_image)
        cv2.destroyAllWindows()
        roi_cropped = self.img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] 
 
        # _, old_dist, _ = histogram_equalization(roi_cropped)
        distribution,mean,standard_deviation = get_distribution_stat(roi_cropped)

        self.ROI_plot.axes[1].clear()
        self.ROI_plot.axes[0].imshow(roi_cropped,cmap='gray',norm= Normalize(vmin=0, vmax=255, clip=True))
        self.ROI_plot.axes[1].bar(np.arange(0,256),distribution)

        self.ui.verticalLayout_equalize.addWidget(self.ROI_plot)
        self.apply_histogram_equalization()
        self.ui.label_viewer_info.setText('mean = {:.1f} standard deviation = {:.1f}'.format(mean,standard_deviation))

        self.ui.tabWidget.setCurrentIndex(2)
  
    def apply_back_projection(self):
        filter = self.ui.comboBox_back_projection_filter.currentText().lower()
        if filter == 'none':
            filter = None

        angle_precision = int(self.ui.comboBox_angle_precision.currentText())
        
        theta = []
        angle = 0
        while angle <180:
            theta.append(angle)
            angle += angle_precision

        print(theta)
        phantom = shepp_logan(256)
        sinogram, dx,dy, reconstruction_fbp = back_projection(phantom,theta, filter)
        self.back_projection_plot.axes[0].imshow(phantom,cmap=plt.cm.Greys_r)
        
        self.back_projection_plot.axes[1].set_title("Radon transform\n(Sinogram)")
        self.back_projection_plot.axes[1].set_xlabel("Projection angle (deg)")
        self.back_projection_plot.axes[1].set_ylabel("Projection position (pixels)")
        self.back_projection_plot.axes[1].imshow(sinogram, cmap=plt.cm.Greys_r,
        extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
        aspect='auto')

        self.back_projection_plot.axes[2].imshow(reconstruction_fbp,cmap=plt.cm.Greys_r)

        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32) 

    def plot_binary_img(self):
        self.img =rgb2gray(cv2.imread('finger_print.png'))
        self.morphology_plot.axes[0].imshow(self.img,cmap='gray')
        self.morphology_plot.axes[1].imshow(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)
    
    def apply_erosion(self):
        self.img = erode(self.img,self.morphology_test_kernel)
        self.morphology_plot.axes[1].imshow(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)       

    def apply_dilation(self):
        self.img = dilate(self.img,self.morphology_test_kernel)
        self.morphology_plot.axes[1].imshow(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32) 

    def apply_openning(self):
        self.img = open_img(self.img,self.morphology_test_kernel)
        self.morphology_plot.axes[1].imshow(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)

    def apply_closing(self):
        self.img = close_img(self.img,self.morphology_test_kernel)
        self.morphology_plot.axes[1].imshow(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32)

    def apply_morphological_filter(self):
        self.img = morphological_filter(self.img)
        self.morphology_plot.axes[1].imshow(self.img,cmap='gray')
        self.resize(self.frameGeometry().width(),self.frameGeometry().height()-32) 




def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()