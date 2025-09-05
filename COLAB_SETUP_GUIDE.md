# 🚀 Google Colab Pro Setup Guide

## 📋 **Quick Setup Steps**

### 1. **Sign up for Google Colab Pro**
- Go to: https://colab.research.google.com/
- Click "Upgrade to Pro" ($10/month)
- Choose Pro plan

### 2. **Enable GPU Runtime**
- In Colab, go to: **Runtime → Change runtime type**
- Set Hardware accelerator to: **T4 GPU** or **V100 GPU**
- Set Runtime shape to: **High-RAM** (16GB)

### 3. **Upload the Training Notebook**
- Download `colab_training_notebook.ipynb` from this repo
- Upload it to Google Colab
- Or copy the notebook content directly

### 4. **Run the Training**
- Execute each cell in order
- Training will take **4-5 hours**
- Expected accuracy: **85-95%**

## ⚙️ **System Requirements**

| Component | Requirement |
|-----------|-------------|
| **RAM** | 16GB (High-RAM runtime) |
| **GPU** | T4 or V100 (16GB VRAM) |
| **Storage** | 100GB (Colab Pro) |
| **Time** | 4-5 hours training |

## 📊 **Expected Results**

With your **6,222 example dataset**:

| Metric | Expected Value |
|--------|----------------|
| **Training Time** | 4-5 hours |
| **Final Accuracy** | 85-95% |
| **Training Loss** | < 1.5 |
| **SQL Validity** | 90%+ |

## 🎯 **Training Configuration**

```bash
python3 train_with_csv.py \
  --csv_file query_data.csv \
  --max_steps 2000 \
  --learning_rate 1e-4 \
  --output_dir ./colab_financial_model
```

## 💰 **Cost Estimate**

- **Google Colab Pro**: $10/month
- **Training Session**: 4-5 hours ≈ $2-3
- **Total Cost**: Very affordable!

## 🚨 **Important Notes**

1. **Keep Colab Active**: Don't let it disconnect during training
2. **Monitor Progress**: Check training logs regularly
3. **Download Model**: Save your trained model before session ends
4. **GPU Memory**: Colab Pro provides sufficient memory for Llama 3.1

## 🎉 **After Training**

Your trained model will be saved as `colab_financial_model.zip` and can be:
- Downloaded to your local machine
- Used for financial text-to-SQL tasks
- Deployed in production applications

## 🔧 **Troubleshooting**

**If training fails:**
- Check GPU is enabled (T4/V100)
- Verify High-RAM runtime is selected
- Ensure Hugging Face token is valid
- Check internet connection is stable

**If memory issues occur:**
- Reduce `max_steps` to 1000
- Use `train_large_dataset.py` instead
- Sample fewer examples (e.g., 3000 instead of 6222)

## 📞 **Support**

If you encounter issues:
1. Check the training logs for error messages
2. Verify all dependencies are installed
3. Ensure your Hugging Face token has Llama access
4. Try reducing the dataset size if memory is limited

---

**Ready to train your financial text-to-SQL model! 🚀**