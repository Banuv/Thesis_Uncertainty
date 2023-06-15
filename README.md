**Uncertainty Quantification of Robot Inverse Kinematics with Neural Networks
**

In motion control of the robotic arm, it is computationally quite complicated to find
an accurate and reliable solution for inverse kinematics. Hence we develop a
Neural Network(NN) model that estimates the IK values as well as the uncertainty
of each estimation. Even though NNs have proven to give highly accurate
predictions and have reached many milestones, it also tends to produce
over-confident predictions at certain times. Therefore, it becomes imperative for the
model to know its unreliability when facing unknown domains.
An uncertainty assessment is essential because we can determine the shortcomings,
and the model can be calibrated for a better quality of predictions. There is quite a
few robotic research showing that estimating uncertainty is profitable. However,
there is very little research comparative study on which of the uncertainty
technique performs better. Another core challenge we will be working on here is
the out-of-distribution(OOD) detection problem, where we would check if the
model can produce a higher uncertainty value for anomalous input or even
co-variate shifts concerning their training data.
The main goal of the thesis is to estimate the uncertainty in the Inverse kinematics
prediction model. We would initially analyse different Uncertainty quantification
techniques such as Deep Ensembles, MC Dropout and MC DropConnect on the
Inverse Kinematic prediction neural network. This evaluation will also include
tuning hyper-parameters and conducting various experiments to understand the
data we have and the model better. We should also be able to infer the optimal
technique for this robotic application. Further on, we also analyse uncertainty
exclusively in the case of OOD data. As a part of the OOD experiment, we apply
various types of split on the dataset and train the NN with partial data, thus
proving higher uncertainty in OOD areas while testing. Finally, being able to
accurately estimate the uncertainty of the deep learning model for robotic
applications.

Find the PDF attached!
[Uncertainty Quantification of Robot IK_Master Thesis Document_BanumathyM.pdf](https://github.com/Banuv/Thesis_Uncertainty/files/11756901/Uncertainty.Quantification.of.Robot.IK_Master.Thesis.Document_BanumathyM.pdf)
