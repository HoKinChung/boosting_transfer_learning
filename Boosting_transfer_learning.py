#! /usr/bin/python2
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
import scipy.io as sio

MAX_ALPHA = 0.5 * np.log((1.0 - 1e-6) / 1e-6)


def read_data(subject, section, seed, pca):
    '''
	#Read the PSD feature of one section of one subject
	Input
	1.subject: subject number, int
	2.section: section number, int, 0 or 1.
	3.seed: random shuffle, int
	4.pca: n components, int
	Ouput 
	1.data:PSD feature data, <np.array>, size(num_samples,120)
	2.label: label, <np.array>, size(num_samples)
	'''
    #Reading and reshaping the awake data
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
    #Concatenate fatigue data and awake data
    data = np.concatenate((awake_psd, fatigue_psd), axis=0)
    label = np.concatenate((awake_label, fatigue_label), axis=0)
    stack = np.column_stack([data, label])
    np.random.seed(seed)
    np.random.shuffle(stack)
    data = stack[:, 0:d1 * d2]
    label = stack[:, -1]
    #Reduce the mean
    mean = np.mean(data, axis=0)
    data = data - mean
    #Use PCA to reduce the number of dimensions
    pca = PCA(n_components=pca)
    data = pca.fit_transform(data)
    return np.array(data), np.array(label)


def separate_target_data(target_data, target_label, N):
    '''
	Extract N samples from target data per class to train
	The rest of target data are used to test
	Output 
	1.target_training_data
	2.target_training_label
	3.target_test_data
	4.target_test_label
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


def weak_svm_train(source_data, source_label, source_weights, target_data,
                   target_label, target_weights):
    '''
	Output
	1.weak SVM model 
	2.weak SVM model's coeffient: alpha
	'''
    #Combined set
    combined_data = np.concatenate((source_data, target_data), axis=0)
    combined_label = np.concatenate((source_label, target_label), axis=0)
    combined_weights = np.concatenate((source_weights, target_weights), axis=0)
    #Train
    model = svm.SVC(C=2**10, kernel='linear')
    model.fit(combined_data, combined_label, sample_weight=combined_weights)
    #Test over target data
    target_weights = target_weights / sum(target_weights)
    result_target_weak_svm = model.predict(target_data)
    #Error with weights
    error = np.multiply(result_target_weak_svm != target_label, target_weights)
    error_weak_svm = error.sum()
    alpha = 0.5 * np.log((1 - error_weak_svm) / max(error_weak_svm, 1e-6))
    return model, alpha


def strong_classifier(input_data, weak_classifiers_set, alpha_set):
    '''
	Input:
	input_data: data need to be predicted, size[num_samples x num_features]
	weak_classifiers_set: a set of weak classifiers,[{w_cla1},{w_cal2},...,{w_clan}]
	Output:
	prediction: the final prediction of input_data, <np.array>,shape(num_sample)
	'''
    num_samples = np.shape(input_data)[0]
    prediction = np.zeros((num_samples))
    for i in range(len(alpha_set)):
        prediction += (alpha_set[i] *
                       weak_classifiers_set[i].predict(input_data))
    prediction = np.sign(prediction)
    return prediction.reshape((num_samples))


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


def MultiSourceTrAdaboost(source_data,
                          source_label,
                          target_data,
                          target_label,
                          M=10):
    '''
	Input 
	1.source_data: source training data, list(<np.array>),[source1,source2,...,sourceN],size(num_samples x num_features)
	2.source_label: source training label,list(<np.array>),size(num_samples)
	3.target_data: target training data, <np.array>, size(num_samples x num_features)
	4.target_label: target training label, <np.array>, size(num_target_sample)
	5.M: maximum number of iteations
	Output
	weak_classifiers_set: a set of weak classifiers
	'''
    N = len(source_label)
    #Initialize the weight vector
    num_source_sample = 0.0
    source_weights = []
    for num_source in range(len(source_data)):
        temp_num = len(source_label[num_source])
        num_source_sample += temp_num
        temp_weights = np.ones((temp_num)) / temp_num
        source_weights.append(temp_weights)

    num_target_sample = target_data.shape[0]
    target_weights = np.ones((num_target_sample)) / num_target_sample

    #Set the constant alpha_S
    alpha_S = 0.5 * np.log(1 + np.sqrt(2 * np.log(num_source_sample / M)))
    #The set of weak classifiers
    weak_classifiers_set = []
    alpha_set = []
    for i in range(M):
        sum_weights = 0.0
        for j in range(N):
            sum_weights += source_weights[j].sum()
        sum_weights = sum_weights + target_weights.sum()
        for j in range(N):
            source_weights[j] = source_weights[j] / sum_weights
        target_weights = target_weights / sum_weights
        #for all N source domains
        for k in range(N):
            max_alpha = np.inf * -1.0
            #Find the weak classifier(svm's parameter) and its error
            model,alpha = weak_svm_train(source_data[k],source_label[k],\
            source_weights[k],target_data,target_label,target_weights)
            #Find the weak classifier with minimum error
            if alpha > max_alpha:
                best_model = model
                best_alpha = alpha
            if alpha > MAX_ALPHA:
                continue
        weak_classifiers_set.append(best_model)
        alpha_set.append(best_alpha)
        #Update the weight vector
        #Update the source weight vector
        for j in range(N):
            prediction = best_model.predict(source_data[j])
            expon_source\
             = -1.0 * alpha_S * np.abs(prediction-source_label[j])
            source_weights[j]\
              = np.multiply(source_weights[j],np.exp(expon_source))
        #Update the target weight vector
        target_prediction = best_model.predict(target_data)
        expon_target = alpha * np.abs(target_prediction - target_label)
        target_weights = np.multiply(target_weights, np.exp(expon_target))
    return weak_classifiers_set, alpha_set


if __name__ == "__main__":
	######################
	# Session to session #
	######################
	#PCA n compoent from 10 to 120
	for pca in range(10, 120, 10):
		#Record the final accuracy of of ten kind of precentage of ten subjects, size(num_precentage,num_subject)
		plot_svm = np.zeros((10, 10))
		plot_transfer = np.zeros((10, 10))
		#Record the standard deviation of svm and MultisourceTrAdaboost
		standard_deviation_svm = np.zeros((10, 10))
		standard_deviation_transfer = np.zeros((10, 10))
		#Percentage from 1 to 10
		for precentage in range(1, 11):
			print '%' * 25, 'precentage: %f' % (precentage * 0.01), '%' * 25
			#Record the accuracy of svm and transfer learning in a precentage, size(num_trials,num_subject)
			trials = 10  # repeat the experiment 'trials' times
			final_acc_svm = np.zeros((trials, 10))
			final_acc_transfer = np.zeros((trials, 10))
			final_num_great = 0
			final_good_subject = []
			final_bad_subject = []
			for tr in range(trials):
				print 'Trials number: ', tr
				#The number that transfer learning is greater than svm
				num_great = 0
				good_subject = []
				bad_subject = []
				for subject in range(10):
					#Read the target data , each target training data has N/2.
					print '*' * 60
					target_data, target_label = read_data(subject, 1, tr, pca)
					print 'Number of target data: ', len(target_label)
					#N:Number of target training data per class
					N = int(np.shape(target_label)[0] * precentage * 0.01)
					target_training_data , target_training_label,\
					target_test_data , target_test_label =\
					separate_target_data(target_data,target_label,N)
					#Read the source data
					source_data_total, source_label_total = read_data(
						subject, 0, tr, pca)
					max_acc = 0.0
					#Transfer learning
					#for NS in range(10,60,1):#For the MultiSourceTrAdaboost
					for NS in range(1, 2, 1):  #For the TrAdaboost
						source_data, source_label = [], []
						ave = len(source_label_total) / NS
						for i in range(NS):
							temp_data = source_data_total[ave * i:ave * (i + 1), :]
							temp_label = source_label_total[ave * i:ave * (i + 1)]
							#If there is only one class ,then reject
							if len(np.unique(temp_label)) == 1:
								continue
							source_data.append(temp_data)
							source_label.append(temp_label)
						M = 10  #The maximum number of iteration
						#Training
						weak_classifiers_set , alpha_set =\
						MultiSourceTrAdaboost(source_data,source_label,\
						target_training_data,target_training_label,M)
						#Test
						prediction = strong_classifier(target_test_data,
													weak_classifiers_set,
													alpha_set)
						error_rate = cal_error_rate(prediction, target_test_label)
						#Choose the one with the highest accuracy
						if max_acc < 1.0 - error_rate:
							max_acc = 1.0 - error_rate
					#Record the accuracy of a trial of a subject
					final_acc_transfer[tr, subject] = max_acc
					print('Accuracy of transfer, subject %d : %f' %
						(subject, max_acc))
					#Baseline
					baseline_training_data =\
					np.concatenate((source_data_total,target_training_data),axis=0)
					baseline_training_label =\
					np.concatenate((source_label_total,target_training_label),axis=0)

					svm_comparsion = svm.SVC(C=2**10, gamma=2**-1, kernel='rbf')
					svm_comparsion.fit(baseline_training_data,
									baseline_training_label)
					svm_acc = svm_comparsion.score(target_test_data,
												target_test_label)
					#Record the accuracy of a trial of a subject
					final_acc_svm[tr, subject] = svm_acc
					print 'Accuracy of SVM: ', svm_acc
					if svm_acc < max_acc:
						num_great += 1
						good_subject.append(subject)
					else:
						bad_subject.append(subject)
				print 'The number transfer learning is greater than svm: ', num_great
				print 'good transfer:', good_subject
				print 'bad transfer: ', bad_subject
			#calculate standard deviation of ten subject in a precentage
			final_std_transfer = np.std(final_acc_transfer, axis=0)
			final_std_svm = np.std(final_acc_svm, axis=0)
			#calculate average accuracy of ten subject in a precentage
			final_acc_transfer = np.mean(final_acc_transfer, axis=0)
			final_acc_svm = np.mean(final_acc_svm, axis=0)
			print '*' * 15, 'Final result of precentage %.3f' % (precentage *
																0.01), '*' * 15
			for i in range(10):
				print ('Final accuracy of transfer subject %d: %f'\
				%(i,final_acc_transfer[i]))
				print 'Final accuracy of SVM: ', final_acc_svm[i]
				print 'Standard Deviation of transfer: ', final_std_transfer[i]
				print 'Standard Deviation of SVM: ', final_std_svm[i]
				if final_acc_svm[i] < final_acc_transfer[i]:
					final_good_subject.append(i)
					final_num_great += 1
				else:
					final_bad_subject.append(i)
				print '-' * 40
			print 'Final number transfer learning is greater than svm: ', final_num_great
			print 'Final good subject: ', final_good_subject
			print 'Final bad subject: ', final_bad_subject

			#Record accuracy and standard deviation of ten subject in a precentage
			plot_svm[precentage - 1, :] = final_acc_svm
			plot_transfer[precentage - 1, :] = final_acc_transfer
			standard_deviation_transfer[precentage - 1, :] = final_std_transfer
			standard_deviation_svm[precentage - 1, :] = final_std_svm

		#Save the average accuracy data to file
		np.savetxt('./paper_result/Acc_svm_C_10_gamma_-1_PCA_%d_9-10.txt' % (pca),
				plot_svm,
				fmt='%.6e')
		np.savetxt('./paper_result/Acc_OneSou_C_10_gamma_-1_PCA_%d_9-10.txt' %
				(pca),
				plot_transfer,
				fmt='%.6e')
		np.savetxt('./paper_result/Std_svm_C_10_gamma_-1_PCA_%d_9-10.txt' % (pca),
				standard_deviation_svm,
				fmt='%.6e')
		np.savetxt('./paper_result/Std_OneSou_C_10_gamma_-1_PCA_%d_9-10.txt' %
				(pca),
				standard_deviation_transfer,
				fmt='%.6e')
