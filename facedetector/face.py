import sys
import time
import numpy as np
import tensorflow as tf
import cv2

import dlib
from facedetector.myutils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
from facedetector.utils import label_map_util
"""
For Import Libery:-

==>from facedetector.face import FaceDetector

For Initialized face detector:-

=>for initialize we need to pass three arg
        1.weight file path of tensorflow model  
        2.label_map file path of tensorflow model
        3.shape_predictor_68_face_landmarks.dat file path
        -----
        4.this arg is optional if you want to change default treshold that is set to 0.7
        you can pass as 4th arg

example:-

detector=FaceDetector(model_path,label_path,threshold(optional))

"""



class FaceDetector:

    def __init__(self,model_path,label_map,align_model,treshold=0.7):

        print("[INFO]Loading model......")
        self.predictor = dlib.shape_predictor(align_model)
        PATH_TO_CKPT =model_path
        PATH_TO_LABELS=label_map
        self.treshold=treshold
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print("[INFO]Done......")
    def __load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)



    def __box_normal_to_pixel(self,boxes,dim):

        box2=[]
        
        for _,j in  enumerate(boxes):
            height, width = dim[0], dim[1]
                #print('--',j)
            box_pixel = [int(j[0]*height), int(j[1]*width), int(j[2]*height), int(j[3]*width)]
                #print('---',box_pixel)
            box2.append(box_pixel)
        return np.array(box2) 
        
    def get_cordinate(self,image,align=False):
        """
        this method is use for getting cordinate of face and get align face
        args:-
        1.image file (array)
        2.(optional)if you want also cordinate with align image then set align falg to True

        this metod return dictonary of cordinate of face and align image(if set align flag True)  
        """
        final_output={}
        self.image=image
        dim=self.image.shape[0],self.image.shape[1]
        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=self.detection_graph, config=config) as sess:
                image_np = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
                start_time = time.time()
                (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
                elapsed_time = time.time() - start_time
                print("Time to predict detection is:--",elapsed_time)
                boxes=np.squeeze(boxes)
                classes =np.squeeze(classes)
                scores = np.squeeze(scores)
                #print("boxes:-",boxes)
      
                boxes=self.box_normal_to_pixel(boxes,dim)
                #print("---",boxes)
                final_box=[]
                for i,j in enumerate(boxes):
                    if scores[i]>self.treshold:
                       # print('face_found')
                        final_box.append(j)

                self.new_boxes=[]
                for i in final_box:
                    self.new_boxes.append([i[1],i[0],i[3],i[2]])
                #print(self.new_boxes)
                if align==False:
                    final_output={'cordinate':self.new_boxes}
                    return final_output
                else:
                    s_height,s_width=image.shape[:2]
                    boxes=[]
                    for i in self.new_boxes:
                        boxes.append(dlib.rectangle(left=i[0], top=i[1], right=i[2], bottom=i[3]))
                    img=[]
                    for i, det in enumerate(boxes):
                        shape = self.predictor(image , det)
                        left_eye = extract_left_eye_center(shape)
                        right_eye = extract_right_eye_center(shape)

                        M = get_rotation_matrix(left_eye, right_eye)
                        rotated = cv2.warpAffine(image, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

                        cropped = crop_image(rotated, det)
                        img.append(cropped)
                    final_output={'cordinate':self.new_boxes,'align_image':img}
                    return final_output                 
    
    


                

      



