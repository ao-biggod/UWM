#\!/bin/bash
# AutoDL DP Baseline: Step 00 - Environment Check
set -e

LOG_FILE="logs/phase1_check.log"
mkdir -p logs

echo "==========================================" | tee -a $LOG_FILE
echo " Phase 0: Environment Check" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "[1/7] pwd" | tee -a $LOG_FILE
pwd | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "[2/7] nvidia-smi" | tee -a $LOG_FILE
nvidia-smi | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "[3/7] conda --version" | tee -a $LOG_FILE
conda --version 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "[4/7] python --version" | tee -a $LOG_FILE
python --version 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "[5/7] df -h" | tee -a $LOG_FILE
df -h | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "[6/7] free -h" | tee -a $LOG_FILE
free -h | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "[7/7] directory listing" | tee -a $LOG_FILE
ls -la | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "  diffusion_policy-main:" | tee -a $LOG_FILE
ls -la diffusion_policy-main/ | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "  data/pusht:" | tee -a $LOG_FILE
ls -la diffusion_policy-main/data/pusht/ 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo " Check complete: $LOG_FILE" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
