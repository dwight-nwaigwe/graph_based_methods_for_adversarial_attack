# graph_based_methods_for_adversarial_attack
This repository contains source code to implement the methods in the article "Graph-based methods coupled with specific distributional distances for adversarial attack detection". The code here pertains to Model1, and Inceptionresnetv2/Mobilenet/VGG19. Source code for Model1 has it's own folder, while that for Inceptionresnetv2/Mobilenet/VGG19 has a different one. To use the source code, you first need your models and adversarial attacks. Adverarial attacks can be made using scripts provided which use the Adverarial Robustness Toolbox (ART). 

For each model, the source code is separated into subdirectories for logistic regression and the statistical tests (WSR). Once you have a model and the adversarial examples, you will need to change file paths so that the code runs. In addition, you will need several packages as shown by the import statements in each file. 

Making the adversarial examples for Model 1 and running the corresponding logistic regression and statistical test source code is simple, but doing this for Inceptionresnetv2/Mobilenet/VGG19 requires a lot more RAM and time; you may need access to a high performance cluster. 

Feel free to contact me with comments or requests for help.
