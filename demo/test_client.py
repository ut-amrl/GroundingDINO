import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
import rospy
from std_msgs.msg import Float32MultiArray
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.msg import *
from amrl_msgs.srv import SemanticObjectDetectionSrv, SemanticObjectDetectionSrvRequest, SemanticObjectDetectionSrvResponse
from sensor_msgs.msg import Image

def numpy_to_ros_image(np_image, encoding="rgb8"):
    """ Convert a NumPy image (OpenCV) to a ROS Image message. """
    ros_image = Image()
    ros_image.height = np_image.shape[0]
    ros_image.width = np_image.shape[1]
    ros_image.encoding = encoding  # "bgr8" for OpenCV images, "rgb8" for PIL images
    ros_image.is_bigendian = 0
    ros_image.step = np_image.shape[1] * np_image.shape[2]  # width * channels
    ros_image.data = np_image.tobytes()
    return ros_image

def call_grounding_dino_service(image_path, query_text):
    rospy.wait_for_service('grounding_dino_bbox_detector')
    
    try:
        # Create a service proxy
        grounding_dino = rospy.ServiceProxy('grounding_dino_bbox_detector', SemanticObjectDetectionSrv)

        # Read image using OpenCV
        cv_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert OpenCV image (NumPy) to ROS Image
        ros_image = numpy_to_ros_image(cv_image)

        # Create request
        request = SemanticObjectDetectionSrvRequest()
        request.query_text = query_text
        request.query_image = ros_image

        # Call the service
        response = grounding_dino(request)

        # Print response (bounding boxes)
        rospy.loginfo(f"Bounding Boxes: {response.bounding_boxes}")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        
if __name__ == "__main__":
    rospy.init_node("grounding_dino_bbox_detector_client")
    # Example usage
    image_path = "cup.png"  # Replace with a valid image path
    query_text = "cup"
    
    call_grounding_dino_service(image_path, query_text)