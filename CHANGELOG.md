# Sep 13
Changed the LL arrival threshold from 0.02 to 0.09 because else it had too many difficulties reaching goals, particularly when there were walls nearby

# Oct 11
Update the cognitive map's current position only when the ratio of thresholds is <0.8, otherwise too many spurious edges get created when the RE's not sure
-> problem: what if we have a decent RE and nodes are close to each other?
