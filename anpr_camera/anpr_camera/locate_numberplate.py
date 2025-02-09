import yolov5
import cv2
import easyocr

easyocr_reader = easyocr.Reader(['en'])

# load model
model = yolov5.load('keremberke/yolov5m-license-plate')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img = 'number_plates/e0d69514813945b9bac22c02212fafbd.jpg'
img = cv2.imread(img)

# perform inference
results = model(img, size=640)

# inference with test time augmentation
results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# cropds the image to just the numberplate
print(f'Found {len(boxes)} numberplates')
for i in range(len(boxes)):
    
    x1, y1, x2, y2 = boxes[i]
    numberplate = img[int(y1):int(y2), int(x1):int(x2)]
    
    print(easyocr_reader.readtext(numberplate, detail=0))

    # saves the image
    cv2.imwrite(f'number_plates/numberplate_{i}.jpg', numberplate)


# # show detection bounding boxes on image
# results.show()

# # save results into "results/" folder
# results.save(save_dir='results/')