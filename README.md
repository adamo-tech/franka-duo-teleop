# Duo teleop

1. Install [ZED SDK](https://docs.stereolabs.com/docs/development/zed-sdk/linux), [ZED GStreamer](https://github.com/stereolabs/zed-gstreamer#linux-installation), and [uv](https://docs.astral.sh/uv/getting-started/installation/) following their instructions.

2. On a fresh Ubuntu PC, install GStreamer's runtime plugins and the Intel hardware encoder. The existing robot already has these installed.

   ```bash
   sudo apt install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav intel-media-va-driver

3. Go to operate.adamohq.com to sign up and get your API key from the settings page.
   ```

4. From the supplied folder, run:

   ```bash
   ADAMO_API_KEY='your-api-key' uv run adamo_real_teleop.py --name duo-real --collision-force 60 --collision-torque 40 --max-reach 0.25
   ```

Starts arm control, cameras, and rumble. `uv` installs all Python dependencies automatically.

