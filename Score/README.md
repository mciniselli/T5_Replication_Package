
In this step we want to use the **score** function.
This function returns a value from -infinity to 0 that is the log likelihood of the prediction (so if score=0 => log (likelihood)=0 => likelihood = 1, the model is very confident about the prediction).
You can run the ipynb notebook and you can save the score file. The function exports by default also the target file, but you can ignore it.
You have to pass as input to the score function the inputs.txt and predictions.txt file that you used for the evaluation of the model (step 4.5).
It returns the score = the likelihood of the predicted sentence given the input.

Once this is done you can save the score as scores.txt in the output folder and run the following command:

```
python3 src/6_evaluate_score/score_analysis.py --input_path inp/6_evaluate_score --score_path out/6_evaluate_score --score
```

From results we can see that when the model has a confidence greater than 0.9 the prediction is correct more than 90\% of the times.
321k perfect predictions out of 411k (78\% of the total predictions) have a confidence greater than 0.9.
The idea is to suggest a prediction to a developer only when we're almost sure that the prediction is correct.
We can suggest the prediction only when the accuracy is greater than 0.9 with a confidence of 90%.

We then want to analyze the length of the predictions
We took the length of each prediction from the 4th column of each raw_data.csv in 001_RobertaCodeCompletion_MSR/slp-core-exp/ROBERTA. and we saved them as length.txt preserving the order of the datasets.

Then we compute mean and median for each class of confidence, considering only perfect, only non perfect and the whole records.

You can compute that running:

```
python3 src/6_evaluate_score/score_analysis.py --input_path inp/6_evaluate_score --score_path out/6_evaluate_score --length
```

You can see that the length of the classes with higher confidence are greater (around 3 tokens vs around 7 for the class with the lowest confidence).

We did a test to see the percentage of perfect predictions based on the number of tokens.
You can run
```
python3 src/6_evaluate_score/score_analysis.py --input_path inp/6_evaluate_score --score_path out/6_evaluate_score --perfect
```
We finally computed the BLEU score metrics and the Levenshtein distance, in order to compare T5 results with RoBERTa.
You can run
```
python3 src/6_evaluate_score/score_analysis.py --input_path inp/6_evaluate_score --score_path out/6_evaluate_score --metrics
```

In `result_score.txt` you can find for each class of confidence, the number and percentage of perfect predictions. You can also find the Levenshtein distance for non perfect predictions for all classes.
In `result_length.txt` you can find the mean and median length of
1. the tokens correctly predicted
2. the target tokens that were not correctly predicted
3. the predicted tokens that were not correctly predicted
4. the target tokens (correctly or not correctly predicted)
In `result_perfect.txt` you can find the number and percentage of perfect predictions for each number of tokens (to compare this with RoBERTa).
For block level we grouped in bins with a breadth of 5 (as done in RoBERTa paper)
In `result_metrics.txt` you can find results for BLEU score and Levenshtein distance.