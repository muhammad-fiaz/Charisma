# Privacy Notice - Charisma

## NO DATA IS COLLECTED ON OUR SERVERS

**Charisma** is designed with your privacy as the highest priority. This notice explains how your data is handled.

### What Data is Processed?

When you use Charisma, the following data is processed **locally on your machine**:

1. **Personal Information** - Name, age, country, location, hobbies, and other details you provide in the UI
2. **Notion Data** - Your notes, pages, and memories fetched from your Notion workspace
3. **Training Data** - The combination of your personal info and Notion memories used to fine-tune the AI model
4. **Model Files** - The fine-tuned AI models saved to your local disk

### Where is Your Data Stored?

- ✅ **Locally on your computer** - All data processing happens on your machine
- ✅ **Your Notion workspace** - Data is fetched directly from Notion's API to your computer
- ✅ **Optional: HuggingFace Hub** - Only if YOU choose to upload your model (controlled by you)

### Where is Your Data NOT Stored?

- ❌ **NOT on our servers** - We don't have servers collecting your data
- ❌ **NOT in the cloud** - Unless you explicitly upload to HuggingFace
- ❌ **NOT shared with third parties** - Your data stays with you
- ❌ **NOT used for analytics** - No telemetry, no tracking, no analytics

### Data Flow

```
Your Notion Account
        ↓
   (API Request)
        ↓
Your Local Computer ← You control this
        ↓
  (Processing)
        ↓
Your Local Model Files
        ↓
   (Optional)
        ↓
HuggingFace Hub (only if you upload)
```

### API Keys and Tokens

- **Notion API Key** - Stored locally in `charisma.toml` on your computer
- **HuggingFace Token** - Stored locally in `charisma.toml` on your computer
- These are used only to authenticate with respective services
- They are NEVER sent to our servers (we don't have servers!)

### Third-Party Services

Charisma interacts with these services directly from your computer:

1. **Notion API** (notion.so)
   - Purpose: Fetch your notes and memories
   - What's sent: Your API key and requests for your pages
   - Privacy: Covered by [Notion's Privacy Policy](https://www.notion.so/Privacy-Policy-3468d120cf614d4c9014c09f6adc9091)

2. **HuggingFace Hub** (huggingface.co) - Optional
   - Purpose: Upload your trained model (if you choose to)
   - What's sent: Your model files and token
   - Privacy: Covered by [HuggingFace Privacy Policy](https://huggingface.co/privacy)

### Logs and Diagnostics

- **Log files** are created in the `./logs/` directory on your computer
- They contain information about the training process
- They may include error messages and debugging information
- They are stored ONLY on your local machine
- You can delete them at any time

### Your Rights and Control

You have complete control over your data:

✅ **Access** - All your data is on your computer, accessible anytime
✅ **Deletion** - Delete any files, logs, or models whenever you want
✅ **Export** - Your model files are standard format and portable
✅ **Opt-out** - Don't upload to HuggingFace if you want complete local control

### Security Best Practices

To keep your data secure:

1. **Keep your API keys private** - Don't share your `charisma.toml` file
2. **Use secure networks** - Especially when fetching from Notion
3. **Backup your models** - Save copies of your trained models
4. **Review HuggingFace settings** - Set models to "private" if uploading
5. **Secure your computer** - Use encryption, passwords, and security software

### Open Source Transparency

Charisma is open source:

- You can review all the code
- You can verify no data is sent to external servers
- You can modify the code to fit your needs
- You can audit the entire application

### Changes to This Notice

If we make changes to how Charisma works:

- We'll update this notice
- We'll update the version number
- You'll see changes in the GitHub repository
- Major changes will be announced in release notes

### Questions or Concerns?

If you have any questions about privacy or data handling:

- Open an issue: https://github.com/muhammad-fiaz/charisma/issues
- Email: contact@muhammadfiaz.com

### Summary

**Key Points:**
- ✅ 100% local processing
- ✅ No data collection
- ✅ No telemetry or tracking
- ✅ You control everything
- ✅ Open source and transparent

**Your data, your computer, your control.**

---

*Last updated: January 2025*
*Version: 0.1.0*
