#!/usr/bin/env python3
"""
Training runner with multiple persistence options for remote SSH training
"""

import os
import sys
import subprocess
import signal
import time
import argparse
import logging
from datetime import datetime

class PersistentTrainingRunner:
    """Runner for persistent training sessions"""
    
    def __init__(self, script_path="jnm_GAN_AHTR.py", log_dir="training_logs"):
        self.script_path = script_path
        self.log_dir = log_dir
        self.session_name = f"gan_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.log_dir, f"{self.session_name}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_with_tmux(self):
        """Run training using tmux (recommended)"""
        print("🚀 Starting training with tmux...")
        print(f"📋 Session name: {self.session_name}")
        
        # Create tmux session
        tmux_cmd = [
            "tmux", "new-session", "-d", "-s", self.session_name,
            "bash", "-c", 
            f"cd {os.getcwd()} && poetry run python {self.script_path} 2>&1 | tee {self.log_dir}/{self.session_name}_output.log"
        ]
        
        try:
            subprocess.run(tmux_cmd, check=True)
            self.logger.info(f"✅ Training started in tmux session: {self.session_name}")
            print(f"\n📌 To attach to session: tmux attach-session -t {self.session_name}")
            print(f"📌 To detach from session: Ctrl+B, then D")
            print(f"📌 To list sessions: tmux list-sessions")
            print(f"📌 To kill session: tmux kill-session -t {self.session_name}")
            print(f"📁 Logs saved to: {self.log_dir}/{self.session_name}_output.log")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to start tmux session: {e}")
            print("💡 Install tmux: sudo apt-get install tmux")
    
    def run_with_screen(self):
        """Run training using screen (alternative)"""
        print("🚀 Starting training with screen...")
        print(f"📋 Session name: {self.session_name}")
        
        screen_cmd = [
            "screen", "-dmS", self.session_name,
            "bash", "-c",
            f"cd {os.getcwd()} && poetry run python {self.script_path} 2>&1 | tee {self.log_dir}/{self.session_name}_output.log"
        ]
        
        try:
            subprocess.run(screen_cmd, check=True)
            self.logger.info(f"✅ Training started in screen session: {self.session_name}")
            print(f"\n📌 To attach to session: screen -r {self.session_name}")
            print(f"📌 To detach from session: Ctrl+A, then D")
            print(f"📌 To list sessions: screen -list")
            print(f"📌 To kill session: screen -X -S {self.session_name} quit")
            print(f"📁 Logs saved to: {self.log_dir}/{self.session_name}_output.log")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to start screen session: {e}")
            print("💡 Install screen: sudo apt-get install screen")
    
    def run_with_nohup(self):
        """Run training using nohup (basic option)"""
        print("🚀 Starting training with nohup...")
        
        nohup_cmd = f"nohup poetry run python {self.script_path} > {self.log_dir}/{self.session_name}_output.log 2>&1 &"
        
        try:
            process = subprocess.Popen(nohup_cmd, shell=True)
            self.logger.info(f"✅ Training started with nohup, PID: {process.pid}")
            print(f"🆔 Process ID: {process.pid}")
            print(f"📁 Logs saved to: {self.log_dir}/{self.session_name}_output.log")
            print(f"📌 To monitor: tail -f {self.log_dir}/{self.session_name}_output.log")
            print(f"📌 To kill process: kill {process.pid}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start nohup process: {e}")
    
    def run_with_systemd(self):
        """Create systemd service for training (advanced)"""
        service_name = f"gan-training-{self.session_name}"
        service_file = f"/tmp/{service_name}.service"
        
        service_content = f"""[Unit]
Description=GAN Training Service - {self.session_name}
After=network.target

[Service]
Type=simple
User={os.getenv('USER')}
WorkingDirectory={os.getcwd()}
ExecStart=/usr/bin/poetry run python {self.script_path}
Restart=on-failure
RestartSec=10
StandardOutput=append:{self.log_dir}/{self.session_name}_output.log
StandardError=append:{self.log_dir}/{self.session_name}_error.log

[Install]
WantedBy=multi-user.target
"""
        
        try:
            # Write service file
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            print(f"📝 Service file created: {service_file}")
            print(f"\n🔧 To install and start service:")
            print(f"   sudo cp {service_file} /etc/systemd/system/")
            print(f"   sudo systemctl daemon-reload")
            print(f"   sudo systemctl enable {service_name}")
            print(f"   sudo systemctl start {service_name}")
            print(f"\n📌 To monitor service:")
            print(f"   sudo systemctl status {service_name}")
            print(f"   sudo journalctl -u {service_name} -f")
            print(f"\n🛑 To stop service:")
            print(f"   sudo systemctl stop {service_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create systemd service: {e}")

def create_monitoring_script():
    """Create monitoring script for training progress"""
    monitoring_script = """#!/bin/bash
# Training Monitor Script

SESSION_NAME="$1"
LOG_DIR="training_logs"

if [ -z "$SESSION_NAME" ]; then
    echo "Usage: $0 <session_name>"
    echo "Available sessions:"
    tmux list-sessions 2>/dev/null || echo "No tmux sessions found"
    exit 1
fi

echo "🔍 Monitoring training session: $SESSION_NAME"
echo "📁 Log directory: $LOG_DIR"

# Function to show GPU usage
show_gpu_usage() {
    echo "🖥️  GPU Usage:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
}

# Function to show recent logs
show_recent_logs() {
    echo "📋 Recent training logs:"
    find $LOG_DIR -name "*$SESSION_NAME*" -type f -exec tail -20 {} \;
}

# Function to show training progress
show_training_progress() {
    echo "📈 Training Progress:"
    find $LOG_DIR -name "*$SESSION_NAME*" -type f -exec grep -i "epoch\|loss\|speed" {} \; | tail -10
}

# Main monitoring loop
while true; do
    clear
    echo "=========================================="
    echo "🚀 GAN Training Monitor - $(date)"
    echo "=========================================="
    
    show_gpu_usage
    echo ""
    show_training_progress
    echo ""
    show_recent_logs
    
    echo ""
    echo "🔄 Refreshing in 30 seconds... (Ctrl+C to exit)"
    sleep 30
done
"""
    
    with open("periksa/monitor_training.sh", "w") as f:
        f.write(monitoring_script)
    
    os.chmod("periksa/monitor_training.sh", 0o755)
    print("✅ Monitoring script created: periksa/monitor_training.sh")

def main():
    parser = argparse.ArgumentParser(description="Persistent Training Runner")
    parser.add_argument("--method", choices=["tmux", "screen", "nohup", "systemd"], 
                       default="tmux", help="Method to run training")
    parser.add_argument("--script", default="jnm_GAN_AHTR.py", 
                       help="Training script to run")
    
    args = parser.parse_args()
    
    runner = PersistentTrainingRunner(script_path=args.script)
    
    print("🎯 Persistent Training Runner")
    print("="*50)
    
    if args.method == "tmux":
        runner.run_with_tmux()
    elif args.method == "screen":
        runner.run_with_screen()
    elif args.method == "nohup":
        runner.run_with_nohup()
    elif args.method == "systemd":
        runner.run_with_systemd()
    
    # Create monitoring script
    create_monitoring_script()

if __name__ == "__main__":
    main()
