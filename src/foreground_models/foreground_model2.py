from torchvision import models
import torchvision.transforms as T
dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

def fg_bg_sepration(image, source):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    colors_label = np.array([(0, 0, 0),  # 0=bg
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    
    for l in range(0, 21):
        idx = image == l
        r[idx] = colors_label[l, 0]
        g[idx] = colors_label[l, 1]
        b[idx] = colors_label[l, 2]
    rgb = np.stack([r, g, b], axis=2)

    # Load the fg input image
    fg = cv2.imread(source)
    
    # Change the color of fg image to RGB
    # and resize image to match shape of R-band in RGB output map
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    fg = cv2.resize(fg,(r.shape[1],r.shape[0]))
 
    # Create a bg array to hold white pixels
    # with the same size as RGB output map
    bg = 255 * np.ones_like(rgb).astype(np.uint8)

    # Convert uint8 to float
    fg = fg.astype(float)
    bg = bg.astype(float)

    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)

    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7,7),0)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the fg with the alpha matte
    fg = cv2.multiply(alpha, fg)

    # Multiply the bg with ( 1 - alpha )
    bg = cv2.multiply(1.0 - alpha, bg)

    # Add the masked fg and bg
    outImage = cv2.add(fg, bg)

    # Return a normalized output image for display
    return rgb, outImage/255

def segment(net, path):
    img = Image.open(path)
    plt.imshow(img); plt.axis('off'); plt.show()
    # Comment the Resize and CenterCrop for better inference results
#     trf = T.Compose([T.Resize(256), 
#                    T.CenterCrop(224), 
#                    T.ToTensor(), 
#                    T.Normalize(mean = [0.485, 0.456, 0.406], 
#                                std = [0.229, 0.224, 0.225])])
    trf = T.Compose([
                   T.CenterCrop(400),
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = fg_bg_sepration(om, path)[0]
    plt.imshow(rgb); plt.axis('off'); plt.show()

segment(dlab, "examples/image1.png")
segment(dlab, "examples/image6.png")
segment(dlab, "examples/image4.png")