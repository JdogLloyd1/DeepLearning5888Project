Agenda
- Role assignments to completion
- Workflow/Schedule and handoffs 
- Code architecture and interfaces 
- Which model architectures to test, who will run which models 

Completed: Replication of model results by Andrew, Jack to confirm good code 

Results look suspicious though. Short variant looks okay but long variant was erroring out. 
Perhaps Pytorch vs original package? Modified learning rate from baseline and got decent results. 

Action: Angela will double check too and post to GitHub 

Code workflow - Angela, Jonathan? 
1. Standardized block to define a model 
2. Single function run with model to spit out results
3. Auto save results to Drive folder 
4. Upload Drive folders to GitHub 

Models we want to run: 
1. Variations on CNNs - Amine - try to pinpoint specific parameters/hyperparameters that were most impactful
2. Transfer learning
a. Transformers - Angela - e.g. Audio Spectrogram Transformer AST, BEATs/AudioMAE
b. Other architectures - Jack - e.g. VGGish, etc (check to make sure they aren't Transformers) 
3. LSTMs - Jonathan

Comment to address: 
The main weakness is that the background is stronger on neural methods than on non-neural alternatives, whereas the prompt ideally asked for both. 
For final report - add some detail on traditional ML methods to the background section, no implementation needed. Give some background on when RNN/LSTM/Transfer learning became popularized 

For the final presentation and report, please consider keeping the comparison set focused so that the key takeaway remains sharp, and it would be great if you could emphasize which changes actually matter most relative to the original Piczak baseline rather than presenting too many incremental variations.

Presentation chief - Andrew

Everyone needs to talk - schedule a Zoom to record together

Q&A board primary responder - Jonathan 

Final report chief - Andrew - will assign sections to people and bring it all together
