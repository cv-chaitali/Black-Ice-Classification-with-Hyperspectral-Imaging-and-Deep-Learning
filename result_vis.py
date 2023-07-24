from testing import *
"""get the reference from data_vis"""
def Patch(data,height_index,width_index):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch
# load the original image
X, y = loadData(dataset)
height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = windowSize
numComponents = K


def predict_image(model, X, y, PATCH_SIZE):
    height, width = y.shape
    outputs = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                image_patch = Patch(X, i, j, PATCH_SIZE)
                X_test_image = image_patch.reshape(1, *image_patch.shape, 1).astype('float32')
                prediction = model.predict(X_test_image)
                prediction = np.argmax(prediction, axis=1)
                outputs[i][j] = prediction + 1

    return outputs


X_pca, pca = applyPCA(X_new, numComponents=numComponents)
X_padded = padWithZeros(X_pca, PATCH_SIZE // 2)

predicted_image = predict_image(model, X_padded, y, PATCH_SIZE)

# visualising the predicted image
spectral.imshow(predicted_image)
