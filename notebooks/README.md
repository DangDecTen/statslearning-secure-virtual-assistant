# Speaker Recognition Models

Conclusion from the evaluation
- Model: Fine-tuned model
	- [`Nampfiev1995/pvad-speechbrain-ft`](https://huggingface.co/Nampfiev1995/pvad-speechbrain-ft) on Hugging Face.
- Enrolment: Request 2–3 utterances at enrolment (94.8-95.7% accuracy for top-1 identification)
	- If your product supports it, consider progressive/continuous enrollment: start the user at n=1 (accept the 88% accuracy risk to reduce friction), then silently fold in more utterances from real usage over time to move them up the curve — this only works if you also design a way to update centroids incrementally rather than only at enrollment time.
- Verification threshold: Use the minDCF threshold on Vietnam-Celeb. Don't use the EER threshold — it balances FAR and FRR equally, but a security app usually wants FAR far lower than FRR (better to ask a legitimate user to retry than let an impostor in).
	- 0.660/0.638 for pretrained
	- 0.431/0.435 for fine-tuned