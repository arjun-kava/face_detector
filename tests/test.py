import unittest 

from face_detector import *
  
class SimpleTest(unittest.TestCase): 
  
    # Returns True or False. 
    def setUp(self):
        self.d=Face_Detector('./model/frozen_inference_graph_face.pb','./protos/face_label_map.pbtxt')
        self.image="5.jpg"


    def test_positive_cordinate(self):
        """Test that the cordinate are always positive.
            Use a tricky case where numerical errors are common.
        """         


        flag=True
        
        cordinate=self.d.get_cordinate(self.image)
        for i in cordinate:
            for j in i:
                if j<0.0:
                    flag=False
                    break
        self.assertTrue(flag)

    def test_rectangle(self):
        """Test that the cordinate are always
        create rectangle or square
        so it always have height and width none-zero
           
        """   
        flag=True
        cordinate=self.d.get_cordinate(self.image)
        for i in cordinate:
            
            if i[2]==0 or i[3]==0:
                flag=False
        self.assertTrue(flag)

  
if __name__ == '__main__': 
    unittest.main() 