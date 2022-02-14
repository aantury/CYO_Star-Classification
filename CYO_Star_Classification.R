#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#start of script  

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(party)) install.packages("party", repos = "http://cran.us.r-project.org")
if(!require(earth)) install.packages("earth", repos = "http://cran.us.r-project.org")

#-------------------------------------------------------------------------------
#load initial libraries
library(tidyverse)
library(caret)
library(data.table)

#-------------------------------------------------------------------------------

#Original data file was obtained from "https://www.kaggle.com/deepu1109/star-dataset/code" 
#under license "Data files © Original Authors" dataset
#owner Deepraj Baidya
#original data file "6 class csv.csv" 

#-------------------------------------------------------------------------------

#Github repository. this is the location of the dataset "6 class.csv" to be used for this CYO project
#https://raw.githubusercontent.com/aantury/CYO_Star-Classification/main/6_class.csv

#Dataset retrieved from the report author's repository using the methodology introduced during the course 
#(Irizarri, R. Introduction to Data Science, p. 109)

url <- "https://raw.githubusercontent.com/aantury/CYO_Star-Classification/main/6_class.csv"
tmp_filename <- tempfile()
download.file(url, tmp_filename)
star_dat <- read_csv(tmp_filename, show_col_types = FALSE) 
file.remove(tmp_filename)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#1. Data exploration & Data Cleansing

#To view the top 6 rows of star_dat dataset and the column names
head(star_dat)

#To view the number or observations (rows) 
nrow(star_dat)

#Star_type has nominal numbers assigned instead of descriptive names. These will be replaced
#to the actual star type for better interpretation 
#e.g Star type 0 = Brown Dwarf, etc., as specified in the original dataset repository 
star_dat$Star_type <- as.character(star_dat$Star_type)
star_dat["Star_type"][star_dat["Star_type"] == 0] <- "Brown_Dwarf"
star_dat["Star_type"][star_dat["Star_type"] == 1] <- "Red_Dwarf"
star_dat["Star_type"][star_dat["Star_type"] == 2] <- "White_Dwarf"
star_dat["Star_type"][star_dat["Star_type"] == 3] <- "Main_Sequence"
star_dat["Star_type"][star_dat["Star_type"] == 4] <- "Supergiant"
star_dat["Star_type"][star_dat["Star_type"] == 5] <- "Hypergiant" 

#-------------------------------------------------------------------------------

# To view the dataset structure 
str(star_dat)

#-------------------------------------------------------------------------------

#To find the number of occurrences of different star colors we use the table() function
Rank_1 <- table(star_dat$Star_color)
Rank_1 %>% as.data.frame() %>% 
  arrange(desc(Freq))

#the Star_color column has duplicate entries for the same color,
#those repeated colors will be merged and to have the same formatting 
star_dat[star_dat == "Blue-white" | star_dat == "Blue White"| star_dat == "Blue white" |
           star_dat == "Blue-White" ] <- "Blue_White"
star_dat[star_dat == "yellow-white"  ] <- "Yellow_White"
star_dat[star_dat == "white" ] <- "White"
star_dat[star_dat == "yellowish" ] <- "Yellowish"
star_dat[star_dat == "Orange-Red" ] <- "Orange_Red"
star_dat[star_dat == "Pale yellow orange" ] <- "Pale_Yellow_Orange"
star_dat[star_dat == "Yellowish White" ] <- "Yellowish_White"
star_dat[star_dat == "White-Yellow" ] <- "White_Yellow"

#-------------------------------------------------------------------------------

#Similarly, to obtain number of occurrences of the different spectral classes we 
#use the table() function. 
Rank2 <- table(star_dat$Spectral_class) 
Rank2 %>% as.data.frame() %>% 
  arrange(desc(Freq))
#We can see that the Spectral class most frequent in the dataset is the spectral class M

#-------------------------------------------------------------------------------

#Converting the character columns to factors. This will help with the analysis with 
#the models to be used later on. The methodology was adapted from
#https://michaelbarrowman.co.uk/post/convert-all-character-variables-to-factors/
star_dat <- star_dat %>% mutate(across(where(is.character),as_factor))

head(star_dat)

#-------------------------------------------------------------------------------

#To see the different features plotted against Star type wrapped around Spectral Class,
#we can use the boxplot diagram
#Star_type vs Temperature
star_dat %>% ggplot(aes(x = Star_type, y = Temperature, fill = Star_type)) + geom_boxplot() +
  facet_wrap(~Spectral_class, scales = "free", ncol = 4)+
  theme(axis.text.x = element_blank(), legend.position="bottom") 

#Star_type vs Luminosity
star_dat %>% ggplot(aes(x = Star_type, y = Luminosity, fill = Star_type)) + geom_boxplot() +
  facet_wrap(~Spectral_class, scales = "free", ncol = 4) +
  theme(axis.text.x = element_blank(), legend.position="bottom")

#Star_type vs Radius
star_dat %>% ggplot(aes(x = Star_type, y = Radius, fill = Star_type)) + geom_boxplot() +
  facet_wrap(~Spectral_class, scales = "free", ncol = 4) +
  theme(axis.text.x = element_blank(), legend.position="bottom")

#Star_type vs Absolute_magnitude
star_dat %>% ggplot(aes(x = Star_type, y = Absolute_magnitude, fill = Star_type)) + 
  geom_boxplot() +
  facet_wrap(~Spectral_class, scales = "free", ncol = 4) +
  theme(axis.text.x = element_blank(), legend.position="bottom")

#Further information regarding the relationship between the different features
#can be seen later in the correlation section


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#2. Feature Selection
#Correlation 
#a. Converting the star_dat dataset into a matrix. Only the continuous features have 
#been selected
star_dat_cM <- star_dat %>% select(Temperature, Luminosity, Radius, 
                                   Absolute_magnitude)%>% as.matrix() 
star_dat_cM

#b. Creation of correlation matrix using the "cor" function. Only showing 2 decimals to ease viewing
CM <- round(cor(star_dat_cM), 2)
CM
#c. To see whether objects are highly correlated and what are they, we use the "findCorrelation" function 
high_corr <- findCorrelation(CM, cutoff=0.5, exact = TRUE)
high_corr
#As can be seen from the table, both Luminosity and Absolute magnitude are highly 
#correlated with each other, therefore, these features should not be used for analysis

#-------------------------------------------------------------------------------

#In addition to the correlation, it is useful to check whether any of the  features 
#in de dataset have near Zero variance. It is important to find and remove any 
#zero or near-zero variance features as they add no value to the analysis. In this 
#case the "nearZeroVar" function from the caret package is used
star_dat_nzv <- nearZeroVar(star_dat, saveMetrics = TRUE)
star_dat_nzv
#From the table above it can be seen that there are no features that are zero variance

#-------------------------------------------------------------------------------

#Checking whether there are any NA from dataset
sum(is.na(star_dat$Temperature))
sum(is.na(star_dat$Luminosity))
sum(is.na(star_dat$Radius))
sum(is.na(star_dat$Absolute_magnitude))
sum(is.na(star_dat$Star_type))
sum(is.na(star_dat$Star_color))
sum(is.na(star_dat$Spectral_class))
#None of the features have NA values

#-------------------------------------------------------------------------------

#We will use the methodology introduced during the course to obtain the test and 
#train datasets for this CYO project. The "createDataPartition" function of the "caret" package 
#is used to create the train and test sets.

# Test set will be 20% of star_dat dataset. This percentage of partition was chosen 
#because the star_dat dataset is not very large as there are 240 observations
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = star_dat$Star_type, times = 1, p = 0.2, list = FALSE)
train_data <- star_dat[-test_index,]      
test_data <- star_dat[test_index,]      
  

train_data
test_data



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#3. Comparison between different machine learning models using the "train" function 
#of the caret package. The target feature is Star_type, therefore, the Star Type 
#on the test set needs to be converted to factor to be able to use it to test the models

test_data$Star_type <- as.factor(test_data$Star_type)

#The predictors to be used are: Temperature, Radius, Star Color and Spectral Class

#-------------------------------------------------------------------------------

#a. Random Forest using caret method "rf" and the "randomForest" package as explained in 
#https://topepo.github.io/caret/train-models-by-tag.html#Random_Forest


#Training the model using default "trainControl" option
library(randomForest)
set.seed(1, sample.kind="Rounding")
star_rf <- train(Star_type ~ Temperature + Radius +  
                           Star_color + Spectral_class, method = "rf", data=train_data, 
                         metric = "Accuracy", trControl = trainControl())
star_rf

#Model prediction using the test_data
star_prediction_rf <- predict(star_rf, test_data)
star_prediction_rf

#Accuracy from the Confusion Matrix 
confusionMatrix(star_prediction_rf, test_data$Star_type)$overall[["Accuracy"]]

#Accuracy table
Accuracy_results <- tibble(Method = "a. Random Forest", Accuracy = 
                                       confusionMatrix(star_prediction_rf, 
                                                       test_data$Star_type)$overall[["Accuracy"]])
Accuracy_results %>% knitr::kable()

#-------------------------------------------------------------------------------

#b. Conditional Inference Random Forest. Implementation of the random forest and bagging
#ensemble (https://www.rdocumentation.org/packages/partykit/versions/1.2-15/topics/cforest) 
#As explained in https://topepo.github.io/caret/train-models-by-tag.html


#Training the model using the default train control values in cforest_control()
library(party)
set.seed(1, sample.kind="Rounding")
star_cforest <- train(Star_type ~ Temperature + Radius +  
                            Star_color + Spectral_class, method = "cforest", data=train_data, 
                          metric = "Accuracy", controls = cforest_control() )
star_cforest

#Model prediction using the test_data
star_prediction_cforest <- predict(star_cforest, test_data)

star_prediction_cforest

#Confusion Matrix 
confusionMatrix(star_prediction_cforest, test_data$Star_type)$overall[["Accuracy"]]

#Accuracy table
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Method = "b. Conditional Inference Random Forest", Accuracy = 
                                       confusionMatrix(star_prediction_cforest, 
                                                       test_data$Star_type)$overall[["Accuracy"]]))
Accuracy_results %>% knitr::kable()

#-------------------------------------------------------------------------------

#c. Bagged Multivariate Adaptive Regression Splines (MARS) 
#(https://www.rdocumentation.org/packages/caret/versions/6.0-90/topics/bagEarthas 
#As explained in https://topepo.github.io/caret/train-models-by-tag.html


#Training the model setting glm option = null as this is not applicable on this model and would otherwise affect the result
library(earth)
set.seed(1, sample.kind="Rounding")
star_bagEarth <- train(Star_type ~ Temperature + Radius +  
                            Star_color + Spectral_class, method = "bagEarth", data=train_data, 
                          metric = "Accuracy", glm = NULL)

star_bagEarth

#Model prediction using the test_data
star_prediction_bagEarth <- predict(star_bagEarth, test_data)

star_prediction_bagEarth

#Confusion Matrix 
confusionMatrix(star_prediction_bagEarth, test_data$Star_type)$overall[["Accuracy"]]

#Accuracy table
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Method = "c. Bagged MARS Model", Accuracy = 
                                       confusionMatrix(star_prediction_bagEarth, 
                                                       test_data$Star_type)$overall[["Accuracy"]]))
Accuracy_results %>% knitr::kable()

#-------------------------------------------------------------------------------

#d. Neural network with principal component step using the "nnet" package and the penalized 
#multinomial regression found in (https://www.rdocumentation.org/packages/nnet/versions/7.3-17/topics/multinom )
#as explained in https://topepo.github.io/caret/train-models-by-tag.html


#Training the model setting the "maxit" parameter to 1000 iterations in order for the model to converge as the 
#default 100 iterations are not sufficient
library(nnet)
set.seed(1, sample.kind="Rounding")
star_pmr <- train(Star_type ~ Temperature + Radius +  
                    Star_color + Spectral_class, method = "multinom", data=train_data, maxit = 1000)
#The model converged and therefore it seems to have worked correctly.
star_pmr

#Model prediction using the test_data
star_prediction_pmr <- predict(star_pmr, test_data)

#Confusion Matrix 
confusionMatrix(star_prediction_pmr, test_data$Star_type)$overall[["Accuracy"]]

#Accuracy table
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(Method = "d. Penalized multinomial regression", Accuracy = 
                                       confusionMatrix(star_prediction_pmr, 
                                                       test_data$Star_type)$overall[["Accuracy"]]))
Accuracy_results %>% knitr::kable()

#-------------------------------------------------------------------------------

#By looking at the different models used, it can be concluded that the best 
#performing model based on the type of data available for this project and according to 
#the the Accuracy metric, the Penalized Multinomial Regression using neural networks achieved
#an accuracy of 0.9583

#End of script  
#-------------------------------------------------------------------------------   
#-------------------------------------------------------------------------------
