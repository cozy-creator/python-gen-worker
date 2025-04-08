# 🚀 Complete Guide: Hosting & Using a Private Python Package Index with Devpi on a DigitalOcean Droplet

This guide walks you through setting up a private package index using `devpi` on a DigitalOcean droplet, publishing your Python package to it, and installing that package using pip or uv.

---

## 🔧 Step 1: Create and Configure Your Droplet

1. **Login to DigitalOcean** → [https://cloud.digitalocean.com](https://cloud.digitalocean.com)
2. Click **"Create"** → **Droplet**
3. Choose:
   - OS: Ubuntu 22.04 LTS
   - CPU & RAM: At least 2 GB RAM recommended
   - Authentication: Choose **Password** and set a strong root password
   - Hostname: (e.g., `gen-orchestrator-private-index`)
4. Click **"Create Droplet"**

---

## 🛠️ Step 2: SSH Into Your Droplet

```bash
ssh root@your_droplet_ip
```

---

## 🐍 Step 3: Install Devpi and Set Up the Server

### Install Python & pip

```bash
apt update && apt install -y python3 python3-pip
```

### Create a virtual environment for devpi

```bash
python3 -m venv devpi-env
source devpi-env/bin/activate
```

### Install devpi server & client

```bash
pip install devpi-server devpi-client
```

### Initialize devpi storage

```bash
devpi-init --serverdir /opt/devpi
```

### Start devpi server

```bash
devpi-server --serverdir /opt/devpi --host 0.0.0.0 --port 3141 &
```

> You can now access devpi from your browser:
> http\://your\_droplet\_ip:3141

---

## 🔑 Step 4: Set Up Devpi on Your Local Machine

### Install devpi client (if not already installed)

```bash
pip install devpi-client
```

### Connect to your server

```bash
devpi use http://your_droplet_ip:3141
```

### Login as root (first-time password is empty)

```bash
devpi login root --password=''
```

### Set a real password

```bash
devpi user -m root password=yourpassword
```

---

## 📦 Step 5: Create an Index and Set It to Inherit from PyPI

```bash
devpi index -c mydev bases=root/pypi
```

Use it:

```bash
devpi use root/mydev
```

---

## 🧱 Step 6: Build and Upload Your Python Package

Inside your project folder:

### Install build tools

```bash
pip install build
```

### Build your package

```bash
python -m build
```

### Or use UV

```
uv build
```



### Upload it to your devpi index

```bash
devpi login root --password=yourpassword

devpi use http://your_droplet_ip:3141/root/mydev

devpi upload
```

---

## 📥 Step 7: Install Your Package Using pip or uv

### pip

```bash
pip install --trusted-host your_droplet_ip \
  --index-url http://your_droplet_ip:3141/root/mydev/+simple \
  yourpackage
```

### uv

```bash
uv pip install --index-url http://your_droplet_ip:3141/root/mydev/+simple yourpackage
```

> If you see dependency resolution issues, ensure `bases=root/pypi` is set so it can fetch missing packages from PyPI.

---

## 🛡️ Bonus: Auto-activate virtualenv on SSH

Edit your `.bashrc`:

```bash
echo 'source ~/devpi-env/bin/activate' >> ~/.bashrc
```

---

## 🔁 Optional: Restart Devpi Server

If you reboot your droplet:

```bash
source ~/devpi-env/bin/activate
devpi-server --serverdir /opt/devpi --host 0.0.0.0 --port 3141 &
```

You can also create a `systemd` service to auto-start it (let me know if you want that).

---

## ✅ Done!

You now have a fully working private Python package index running on your own server.

