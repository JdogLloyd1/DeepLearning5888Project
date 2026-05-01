**FINISH BEFORE: THURSDAY 9:00 PM EST**

| Checklist | Action Item List  |  |
| :---- | :---- | :---- |
|  | 1 | Jack to upload revised code to include baseline requirements (clip voting) |
| Done | 2 | Jonathan to find the optimal run on which all others will be based (Long, L/P, etc) |
| Andrew \- Done Amine \- Done Angela \- Done Jack \-  Jonathan \-  | 3 | Everyone review their slides \- update and send your final slides to Andrew for final deck |
|  | 4 | Andrew to compile final deck and upload to GitHub |
| Andrew \- Done Amine \- Done Angela \- Done Jack \-  Jonathan \-  | 5 | Build the narrative for your slide in shared google doc |
|  | 7 | Angela to build the shared narrative  |
| Andrew \-  Amine \-  Angela \-  Jack \-  Jonathan \-  | 8 | Everyone \- ensure the architecture table on the slide represents what you actually ran |
| Done | 9 | Angela to double check clip level voting  Answer: Piczak CNN classifies short spectrogram segments and requires majority voting to produce a final clip label (ex. classifies 5 1-second clips and produces final prediction for full 5-second clip based on voting) while AST and BEATs process the full 5-second waveform as a single input to form the prediction. |
| Andrew \-  Amine \- Done Angela \- Done Jack \- Done Jonathan \- Done | 10 | Everyone to submit results and code to git repo |

# Presentation

Slide 1 \- Title Slide  
Presenter: Jonathan

Slide 2 \- Intro  
Presenter: Amine

We aimed to explore how the evolution of neural network architectures translates into measurable improvements in accuracy over time. To do this, we needed a clean, high-quality dataset of manageable size that could serve as a consistent benchmark. Piczak’s work provided exactly that foundation, allowing us to replicate his dataset and task while progressively advancing through different model eras—from CNNs trained from scratch, to transfer learning with pretrained CNNs, and finally to transfer learning using pretrained audio transformers. At each stage, we compared performance against a shared baseline to track the impact of architectural advancements clearly. 

Slide 3 \- Hypothesis and Experiment   
Presenter: Amine

We hypothesized that modern neural network techniques would outperform the original model. To test this, we first re-implemented Piczak’s model to establish a baseline. We then kept the dataset and task consistent while evolving the modeling approach and evaluating performance using the final five-fold test accuracy with clip-level voting applied where appropriate. For consistency, we ultimately relied on results from long audio segments, using probability-based voting as the primary comparator to ensure an apples-to-apples evaluation. This framework was applied across the three neural network “eras” outlined in our introduction.

Slide 4 \- Piczak and ESC-50  
Presenter: Andrew  
Here we’ve got details about the baseline and the dataset.  We chose Piczak’s ESC-50 because we felt his work took a relatively unstudied segment, which was audio sound training, and that would do well to represent the growth in neural network capabilities from that point.  He also had a robust yet compact data set that was well organized – 2000 audio samples, 50 classes, 5 folds – and they were of a manageable size at 5 seconds per clip, which presented a manageable data set to run our multiple different models against.  
Looking specifically at what he did with it, Piczak converted audio clips into log-mel spectrogram segments, classified each segment, and then used clip-level voting to produce one prediction per audio clip. That gave us the fixed baseline we needed 

Slide 5 \- Transition Slide \- first era  
Presenter:  Andrew  
Moving into the first block of models to be reviewed, here we are looking at neural network models that rely solely on the data set for training. That means it has to first learn the features of the audio clips themselves, the acoustic patterns in the spectrogram that would help distinguish classes \- short bursts of sound like a dog bark, repetitive textures like a bird chirp, or changes over time.  Then it has to learn the classifier, meaning mapping those patterns to the labels from ESC-50.  
Within this era we will look at first recreating Piczak’s model to build our baseline, then we run a series of CNN variations before introducing an LSTM extension.

Slide 6 \- Piczak Baseline CNN  
Presenter:  Andrew

Here is where we established the baseline from which subsequent models would be evaluated. We needed to re-create it to ensure we had a solid understanding of how it was created as well as to use that as the springboard for other models. What you'll see in the baseline model accuracy curves is the various runs that were performed, again, recreating the original model. However, it was ultimately the Long-Segment, clip level run that was highlighted by Piczak and will serve as the basis of comparison going forward. We underran slightly at 63.5%, but felt that was close enough to treat as a credible baseline and accurate enough replication of his work.

Slide 6 \- CNN Variations   
Presenter: Amine 

Slide 7 \- LSTM   
Presenter: Jonathan

One major model architecture development was the LSTM \- long short-term memory. This model architecture was designed to remember information for long periods of time and handle sequential data very well. We designed a simple LSTM baseline model that we varied over 4 factors \- hidden layer size, number of LSTM layers, bidirectional training, and pooling. The best iteration of the LSTM models reported a clip-level accuracy lower than the Piczak CNN’s LP result, which was initially a surprising result to the team, especially given the high training accuracy. However, this ESC-50 dataset is based on 5-second clips, which may not be long enough to fully take advantage of the LSTM’s long sequential data strength. For these shorter clips, CNNs are relatively better suited. In addition, LSTMs require a large amount of data to train effectively, and the dataset’s 2,000 recordings were likely not enough to take advantage of the LSTM’s data scaling strength, resulting in the model overfitting to the training set. 

Slide Transition to pretrained CNNs

Presenter: Jack   
As we shifted from from scratch models to transfer learning from pretrained models era of neural networks, the expectation was that we would see a significant improvement in model performance, as the pretrained models can be leveraged to fine-tune learning outcomes on our chosen dataset. The pretrained models used for this comparative study were VGGish and YAMNet as discussed on the next 2 slides. 

Slide 8 \- VGGish

Presenter: Jack 

The listed model set up is as shown on the left, and the models were run for both a fully frozen transfer learning as well as a fine-tuned model with the final layer unfrozen. While both models significantly outperformed the Piczak baseline in training accuracy, but test accuracy suffered as the model seemed to overfit the dataset too much leaving final average test accuracies at 64.5% and 61% for frozen and fine-tuned model set ups respectively.

Slide 9 \- YAMNet  
Presenter: Jack 

While it was predicted that VGGish would perform better \- under the same model set up as YAMNet \- being that VGGish is a larger model, YAMNet performed much better in training and test accuracy, and we assume that is because the model did not overfit the data as much as VGGish did.  

YAMNet’s strides in average training and test accuracy significantly outperform the Piczak model with final average test accuracies of 88.5% and 87.25% respectively for frozen and fine-tuned model runs. 

Slide 10 \- pretrained foundation models for audio  
Presenter: Angela

Up to this point, we’ve already introduced transfer learning.

VGGish and YAMNet were also pretrained on large audio datasets like AudioSet, so the idea of learning audio representations elsewhere is not new.

However, their performance shows that pretraining alone is not enough.

This next shift focuses on what actually changed in modern models: the *quality and scale of the learned representations*.

AST and BEATs still use transfer learning, but they use much larger, more powerful pretrained models with transformer architectures that can capture broader context.

The key is *how well* they learn and represent audio before fine-tuning on ESC-50.

Slide 11 \- AST

Presenter: Angela

AST stands for Audio Spectrogram Transformer.

The easiest way to think about AST is that it keeps the same spectrogram idea as Piczak, but replaces the small CNN trained from scratch with a large pretrained transformer backbone.

Piczak splits the clip into short spectrogram segments, predicts each segment separately, and then uses voting to produce the final clip label.

AST instead processes the full 5-second clip at once.

Because the transformer sees the entire clip, it can model long-range relationships and produce a direct clip-level prediction without needing a voting stage.

This produces a major jump in performance so while Piczak reported 64.5 percent clip-level accuracy, AST achieved 95.25 percent average accuracy across 5 folds.

Slide 12 \- BEATs

Presenter: Angela

If AST alone performed well, we could argue that maybe it is just a better spectrogram model.

BEATs uses a foundation-style pretrained audio encoder trained on large-scale audio data before adaptation to ESC-50.

Instead of learning representations from scratch, it starts with strong audio knowledge already built in. Like AST, it processes the full waveform directly and does not require clip-level voting.

The BEATs model achieved 94.40 percent average accuracy across 5 folds, which is extremely close to AST.

That is more than a 30% improvement, showing that the biggest gains come from modern large-scale representation learning, not just switching model architectures.

Slide 13 \- Conclusion 

Presenter: Jack  
Our results did not show the smooth chronological improvement we initially expected. Instead, the central lesson was that model fit mattered more than model age

CNNs were a strong baseline because spectrogram audio naturally aligns with convolutional feature learning.

Hyperparameter tuning and added architecture complexity helped only a small amount when the model still had to learn from ESC-50 alone, showing the importance of model selection.

The major gains appeared when the model brought in prior audio knowledge from large-scale pretraining.

What went well was the dataset choice: ESC-50 gave us a compact, clean, repeatable benchmark.

What did not go well was reproducibility and comparison discipline. A 2015 baseline was harder to recreate than expected, and small, important details like voting, fold count, and final-versus-best accuracy changed the story.

If we repeated this project, we would lock the evaluation protocol earlier, run fewer variants more deeply, and add runtime per cost as a second performance dimension.