# R2 Troubleshooting Guide

This guide helps resolve common issues when using `doc2lora convert-r2` with Cloudflare R2 buckets.

## Common Errors and Solutions

### 1. "The specified key does not exist" / "NoSuchKey"

**Symptoms:**
```
DEBUG:botocore.parsers:Response body:
b'<?xml version="1.0" encoding="UTF-8"?><Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message></Error>'
❌ Error: 'str' object has no attribute 'get'
```

**Causes & Solutions:**

#### A. Endpoint URL includes bucket name
- **Problem**: Your R2_ENDPOINT_URL accidentally includes the bucket name
- **Fix**: Remove bucket name from endpoint URL

❌ **Wrong:**
```
R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com/my-bucket
```

✅ **Correct:**
```
R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com
```

#### B. Empty bucket or wrong folder prefix
- **Problem**: Bucket has no files or folder prefix is incorrect
- **Fix**: Check bucket contents and folder prefix

```bash
# List bucket contents to verify
doc2lora scan-r2 your-bucket-name --env-file .env
```

### 2. "Bucket does not exist"

**Symptoms:**
```
❌ Error: Bucket 'my-bucket' does not exist
```

**Solutions:**
1. **Check bucket name spelling**
2. **Verify bucket exists in your R2 dashboard**
3. **Ensure credentials have access to this bucket**

### 3. "Access denied"

**Symptoms:**
```
❌ Error: Access denied. Check your credentials and permissions
```

**Solutions:**
1. **Verify R2 credentials are correct**
2. **Check R2 token permissions include Object Read**
3. **Ensure bucket policy allows access**

### 4. "Invalid R2 endpoint URL"

**Symptoms:**
```
❌ Error: Invalid R2 endpoint URL format
```

**Solution:**
Get the correct endpoint from Cloudflare Dashboard:
1. Go to Cloudflare Dashboard > R2
2. Click "Manage R2 API tokens"
3. Copy the "Use with S3 API" endpoint
4. Format: `https://your-account-id.r2.cloudflarestorage.com`

## Correct .env File Format

```env
# R2 Credentials
R2_ACCESS_KEY_ID=your_access_key_here
R2_SECRET_ACCESS_KEY=your_secret_key_here
R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com

# Optional: specify bucket and folder
R2_BUCKET_NAME=my-documents-bucket
R2_FOLDER_PREFIX=training-docs
```

## Testing Your R2 Configuration

### 1. Test Connection
```bash
doc2lora convert-r2 your-bucket-name --env-file .env --output test.json --epochs 1 --batch-size 1
```

### 2. List Files (if scan command exists)
```bash
# This would list files in your bucket
# doc2lora scan-r2 your-bucket-name --env-file .env
```

### 3. Minimal Test with Small Model
```bash
doc2lora convert-r2 your-bucket-name \
  --env-file .env \
  --model microsoft/DialoGPT-small \
  --epochs 1 \
  --batch-size 1 \
  --max-length 256 \
  --output test-adapter.json
```

## Getting R2 Credentials

1. **Login to Cloudflare Dashboard**
2. **Go to R2 Object Storage**
3. **Click "Manage R2 API tokens"**
4. **Create a new token with:**
   - Object Read permissions
   - Object Write permissions (if uploading)
   - Specify bucket restrictions if needed

## Endpoint URL Examples

Different Cloudflare accounts have different formats:

```
# Format 1 (most common)
https://abc123def456.r2.cloudflarestorage.com

# Format 2 (alternative)
https://your-account-name.r2.cloudflarestorage.com
```

**Important:** Never include the bucket name in the endpoint URL!

## Debugging Tips

### Enable Verbose Logging
```bash
doc2lora convert-r2 your-bucket --verbose --env-file .env
```

### Check Bucket Contents
Use the AWS CLI or another S3-compatible tool:
```bash
aws s3 ls s3://your-bucket-name/ --endpoint-url https://your-account.r2.cloudflarestorage.com
```

### Verify Credentials
Test with a simple HEAD request:
```python
import boto3
client = boto3.client('s3',
    endpoint_url='https://your-account.r2.cloudflarestorage.com',
    aws_access_key_id='your-key',
    aws_secret_access_key='your-secret')
client.head_bucket(Bucket='your-bucket')
```

## Still Having Issues?

1. **Check this troubleshooting guide first**
2. **Enable verbose logging with `--verbose`**
3. **Verify your .env file format**
4. **Test with a simple S3 client to isolate the issue**
5. **Check Cloudflare R2 dashboard for bucket permissions**
