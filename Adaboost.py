#! /usr/bin/python2
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA


def read_data(subject, section, seed, n_pca):
    '''
	#Read the PSD feature of one section of one subject
	Input
	1.subject: subject number, int
	2.section: section number, int, 0 or 1.
	3.seed: random select, int
	4.n_pca: n component retain.
	Ouput 
	1.data:PSD feature data, <np.array>, size(num_samples,120)
	2.label: label, <np.array>, size(num_samples)
	'''
    #Reading and reshaping the awake data...
    Awake_PSD = sio.loadmat('./Data/Awake_PSD_norm.mat')
    read_awake_psd = Awake_PSD['all_power_awake_norm'][0, subject][0, section]
    d1 = np.shape(read_awake_psd)[0]
    d2 = np.shape(read_awake_psd)[1]
    d3_a = np.shape(read_awake_psd)[2]
    awake_psd = np.reshape(read_awake_psd, (d1 * d2, d3_a))
    awake_psd = awake_psd.T
    awake_label = np.ones((d3_a))

    #Reading and reshaping the fatigue data
    Awake_PSD = sio.loadmat('./Data/Fatigue_PSD_norm.mat')
    read_fatigue_psd = Awake_PSD['all_power_fatigue_norm'][0, subject][0,
                                                                       section]
    d1 = np.shape(read_fatigue_psd)[0]
    d2 = np.shape(read_fatigue_psd)[1]
    d3_f = np.shape(read_fatigue_psd)[2]
    fatigue_psd = np.reshape(read_fatigue_psd, (d1 * d2, d3_f))
    fatigue_psd = fatigue_psd.T
    fatigue_label = -1.0 * np.ones((d3_f))
    #Concatenate data
    data = np.concatenate((awake_psd, fatigue_psd), axis=0)
    label = np.concatenate((awake_label, fatigue_label), axis=0)
    stack = np.column_stack([data, label])
    np.random.seed(seed)
    np.random.shuffle(stack)
    data = stack[:, 0:d1 * d2]
    label = stack[:, -1]
    #reduce the mean
    mean = np.mean(data, axis=0)
    data = data - mean
    #Use PCA to reduce the dimension of feature space
    pca = PCA(n_components=n_pca)
    data = pca.fit_transform(data)
    return np.array(data), np.array(label)


def separate_target_data(target_data, target_label, N):
    '''
	This function seperate target data into two parts: training data and test data
	N is the number of target data per class
	Output the training data and test data
	'''
    target_training_data = []
    target_training_label = []
    target_test_data = target_data
    target_test_label = target_label
    index_awake = np.argwhere(target_label == 1)[0:N, 0]
    index_fatigue = np.argwhere(target_label == -1)[0:N, 0]
    index = np.concatenate((index_awake, index_fatigue), axis=0)
    for i in index_awake:
        target_training_data.append(target_data[i, :])
        target_training_label.append(target_label[i])
    for i in index_fatigue:
        target_training_data.append(target_data[i, :])
        target_training_label.append(target_label[i])
    target_test_data = np.delete(target_data, index, axis=0)
    target_test_label = np.delete(target_label, index, axis=0)
    return np.array(target_training_data), np.array(
        target_training_label), target_test_data, target_test_label


def cal_error_rate(prediction, ground_truth):
    '''
	Input:
	prediction: prediction labels,<np.array>,shape(num_sample)
	ground_truth: true labels,<np.array>,shape(num_sample)
	Output:
	error_rate: error rate, float
	'''
    error = np.multiply(prediction != ground_truth, np.ones((len(prediction))))
    error_rate = error.sum() / len(prediction)
    return error_rate


def Adaboost_svm(training_data, training_label, test_data, test_label, M=10):
    '''
	The adaboost model whose weak classifier is svm
	This function return the accuracy of model over the test dataset.
	'''

    #Init the weights
    weights = np.ones((len(training_label)))
    #Save the weak classifiers
    set_weak_classifiers = []
    set_alpha = []
    for m in range(M):
        weak_classifier = {}
        weights = weights / sum(weights)
        #Train
        model = svm.SVC(C=2**10, kernel='linear')
        #model = svm.SVC(C=2**10,gamma=2**-1,kernel='rbf')
        model.fit(training_data, training_label, sample_weight=weights)
        #Calculate error and weight
        prediction = model.predict(training_data)
        error = np.multiply(weights, prediction != training_label)
        error_rate = error.sum()
        alpha = 0.5 * np.log((1.0 - error_rate) / max(error_rate, 1e-6))
        #Update the weights
        expon = -1.0 * alpha * np.multiply(training_label, prediction)
        weights = np.multiply(weights, np.exp(expon))
        #Normalization
        weights = weights / weights.sum()
        #Save model and alpha
        set_weak_classifiers.append(model)
        set_alpha.append(alpha)
    test_prediction = np.zeros((len(test_label)))
    for i in range(len(set_alpha)):
        test_prediction += (set_alpha[i] *\
         set_weak_classifiers[i].predict(test_data))
    test_prediction = np.sign(test_prediction)
    test_error = cal_error_rate(test_prediction, test_label)
    return (1.0 - test_error)


if __name__ == "__main__":

    ####################
    #Session to session#
    ####################
    for pca in range(10, 120, 20):
        #Record the final accuracy of of ten kind of precentage of ten subjects, size(num_precentage,num_subject)
        plot_acc_svm = np.zeros((10, 10))
        plot_std_svm = np.zeros((10, 10))
        #Record the standard deviation of Adaboost
        standard_deviation_svm = np.zeros((10, 10))
        #Count the percentage number
        j = 0
        for precentage in range(1, 11, 1):
            print '%' * 25, 'Precentage %f' % (precentage * 0.01), '%' * 25
            trials = 10
            final_acc_svm = np.zeros((trials, 10))
            for tr in range(
                    trials
            ):  #tr is the trial number, it is also the random seed
                print 'Trial number: ', tr
                for subject in range(10):
                    #Read the target data
                    print '*' * 60
                    target_data, target_label = read_data(subject, 0, tr, pca)
                    print 'Number of target data: ', len(target_label)
                    #N:Number of target training data per class
                    N = int(len(target_label) * precentage * 0.01)
                    print 'Number of target training data per class ', N
                    target_training_data , target_training_label,\
                    target_test_data , target_test_label =\
                    separate_target_data(target_data,target_label,N)
                    #Read the source data
                    source_data_total, source_label_total = read_data(
                        subject, 1, tr, pca)
                    #Combine the source data and target training data as training data
                    baseline_training_data =\
                    np.concatenate((source_data_total,target_training_data),axis=0)
                    baseline_training_label =\
                    np.concatenate((source_label_total,target_training_label),axis=0)
                    #Training and evaluate the accuracy on test data
                    adaboost_acc = Adaboost_svm(baseline_training_data\
                    ,baseline_training_label,target_test_data,\
                    target_test_label)

                    final_acc_svm[tr, subject] = adaboost_acc
                    print('Accuracy of Adaboost of Subject %d: %f' %
                          (subject, adaboost_acc))
            #calculate standard deviation of ten subject in a precentage
            final_std_svm = np.std(final_acc_svm, axis=0)
            #calculate average accuracy of ten subject in a precentage
            final_acc_svm = np.mean(final_acc_svm, axis=0)
            print '*' * 15, 'Final result of precentage %.3f' % (
                precentage * 0.01), '*' * 15
            for i in range(10):
                print('Final accuracy of Adaboost subject %d: %f' %
                      (i, final_acc_svm[i]))
                print 'Standard Deviation of SVM: ', final_std_svm[i]
                print '-' * 40

            #Record accuracy and standard deviation of ten subject per precentage
            plot_acc_svm[j, :] = final_acc_svm
            plot_std_svm[j, :] = final_std_svm
            j += 1

        #Save the average accuracy data to file
        np.savetxt('./acc_Adaboost_C_10_gamma_-1_PCA_%d.txt' % (pca),
                   plot_acc_svm,
                   fmt='%.6e')
        np.savetxt('./std_Adaboost_C_10_gamma_-1_PCA_%d.txt' % (pca),
                   plot_std_svm,
                   fmt='%.6e')
