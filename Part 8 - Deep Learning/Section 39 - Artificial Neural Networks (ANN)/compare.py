import datetime
from ann_function import ann
import matplotlib.pyplot as plt

file = open("performace.txt","a+")

file.write("\n" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

performance = []
batch_size_list = []
epoch_list=[]
accuracy_list = []

for batch_size in range(100,1001,100):
    for epoch in range(100,1001,100):
        cm = ann(batch_size, epoch)
        accuracy = (( cm[0][0]+cm[1][1] )/2000)*100
        performance_loop = [batch_size,epoch,accuracy]

        batch_size_list.append(batch_size)
        epoch_list.append(epoch)
        accuracy_list.append(accuracy)

        performance.append(performance_loop)

        string = "\nBatch Size : " + str(performance_loop[0]) + "\tEpochs : " + str(performance_loop[1]) + "\tAccuracy : " + str(performance_loop[2])
        file.write(string)

file.close()

# fig = plt.figure()
#
# # Plotting the graph with the data
# plt.subplot(2, 2, 1)
# plt.plot(batch_size_list, accuracy_list,"r", label = "Batch Size")
#
# plt.subplot(2, 2, 2)
# plt.plot(epoch_list, accuracy_list,"b", label = "Epoch")
#
#
#
# # Generating the legend and the axis and title
# plt.legend(loc='upper right')
# plt.title("Variable Tweaking - Batch Size and Epoch vs Accuracy")
# plt.xlabel("Batch Size and Epochs")
# plt.ylabel("Accuarcy")
# # Saving the graph
# plt.savefig('performance.png')
# # Showing the graph
# plt.show()




