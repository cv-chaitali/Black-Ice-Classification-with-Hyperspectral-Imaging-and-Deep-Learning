from data_vis import *
from data_proc import *
# from model import *
from keras.models import load_model

model = load_model ("your_model_path")

# plt.figure(figsize=(7,7)) 
# plt.grid() 
# plt.plot(history.history['loss'])

# plt.plot(history.history['loss'])
# plt.ylabel('Loss') 
# plt.xlabel('Epochs') 
# plt.legend(['Training','Validation'], loc='upper right') 
# plt.savefig("loss_curve.png") 
# plt.show()

# plt.figure(figsize=(5,5)) 
# plt.ylim(0,1.1) 
# plt.grid() 
# plt.plot(history.history['accuracy'])

# plt.plot(history.history['accuracy'])
# plt.ylabel('Accuracy') 
# plt.xlabel('Epochs') 
# plt.legend(['Training','Validation']) 
# plt.savefig("acc_curve.png") 
# plt.show()



model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
Xtest.shape


ytest = np_utils.to_categorical(ytest)
ytest.shape

Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test, axis=1)

classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
print(classification)


import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score

def AA_andEachClassAccuracy(confusion_matrix):
    diagonal_values = np.diag(confusion_matrix)
    row_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(diagonal_values / row_sum)
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def evaluate_model(model, X_test, y_test):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)

    target_names = ['BlackiceR', 'blackice']
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100

    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100


# Assuming you already have 'model', 'X_test', and 'y_test' available
classification, confusion, Test_Loss, Test_accuracy, oa, each_acc, aa, kappa = evaluate_model(model, X_test, y_test)

print("Classification Report:\n", classification)
print("Confusion Matrix:\n", confusion)
print("Test Loss:", Test_Loss)
print("Test Accuracy:", Test_accuracy)
print("Overall Accuracy:", oa)
print("Each Class Accuracy:", each_acc)
print("Average Accuracy:", aa)
print("Kappa Score:", kappa)

