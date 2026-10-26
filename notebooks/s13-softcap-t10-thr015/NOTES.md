# s13 - s11 with tau 1.0 (temperature sensitivity probe)

tau 1.0 uses the pipeline's probabilities as they are (no sharpening). In the CPU proxy tau 0.5 was better, but the
pipeline's log-probs are sharper than the proxy's, so this pins down which direction tau should move on the real stack.
