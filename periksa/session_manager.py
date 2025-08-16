#!/usr/bin/env python3
"""
Session manager for training sessions
"""

import subprocess
import os
import json
import time
from datetime import datetime
from typing import List, Dict

class SessionManager:
    """Manage training sessions across different persistence methods"""
    
    def __init__(self):
        self.sessions_file = "training_logs/active_sessions.json"
        self.ensure_sessions_file()
    
    def ensure_sessions_file(self):
        """Ensure sessions file exists"""
        os.makedirs("training_logs", exist_ok=True)
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w') as f:
                json.dump([], f)
    
    def list_tmux_sessions(self) -> List[Dict]:
        """List all tmux sessions"""
        try:
            result = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                sessions = []
                for line in result.stdout.strip().split('\n'):
                    if line and 'gan_training' in line:
                        sessions.append({
                            'name': line,
                            'type': 'tmux',
                            'status': 'active'
                        })
                return sessions
        except FileNotFoundError:
            pass
        return []
    
    def list_screen_sessions(self) -> List[Dict]:
        """List all screen sessions"""
        try:
            result = subprocess.run(["screen", "-list"], capture_output=True, text=True)
            sessions = []
            for line in result.stdout.split('\n'):
                if 'gan_training' in line:
                    parts = line.strip().split('\t')
                    if len(parts) > 0:
                        session_info = parts[0].split('.', 1)
                        if len(session_info) > 1:
                            sessions.append({
                                'name': session_info[1],
                                'type': 'screen',
                                'status': 'active' if 'Attached' in line else 'detached'
                            })
            return sessions
        except FileNotFoundError:
            pass
        return []
    
    def get_gpu_usage(self) -> Dict:
        """Get current GPU usage"""
        try:
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpus.append({
                                'index': parts[0],
                                'name': parts[1],
                                'utilization': f"{parts[2]}%",
                                'memory_used': f"{parts[3]} MB",
                                'memory_total': f"{parts[4]} MB",
                                'temperature': f"{parts[5]}°C"
                            })
                return {'gpus': gpus, 'available': True}
        except FileNotFoundError:
            pass
        return {'gpus': [], 'available': False}
    
    def show_status(self):
        """Show comprehensive status of all training sessions"""
        print("🎯 Training Session Status")
        print("=" * 60)
        
        # Show tmux sessions
        tmux_sessions = self.list_tmux_sessions()
        if tmux_sessions:
            print("\n🖥️  TMux Sessions:")
            for session in tmux_sessions:
                print(f"   📋 {session['name']} - {session['status']}")
                print(f"      Attach: tmux attach-session -t {session['name']}")
        
        # Show screen sessions
        screen_sessions = self.list_screen_sessions()
        if screen_sessions:
            print("\n📺 Screen Sessions:")
            for session in screen_sessions:
                print(f"   📋 {session['name']} - {session['status']}")
                print(f"      Attach: screen -r {session['name']}")
        
        # Show GPU usage
        gpu_info = self.get_gpu_usage()
        if gpu_info['available']:
            print("\n🖥️  GPU Usage:")
            for gpu in gpu_info['gpus']:
                print(f"   GPU {gpu['index']} ({gpu['name']}): "
                      f"Util: {gpu['utilization']}, "
                      f"Mem: {gpu['memory_used']}/{gpu['memory_total']}, "
                      f"Temp: {gpu['temperature']}")
        else:
            print("\n⚠️  GPU monitoring not available")
        
        # Show recent logs
        self.show_recent_logs()
    
    def show_recent_logs(self):
        """Show recent training logs"""
        print("\n📋 Recent Training Logs:")
        log_dir = "training_logs"
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
            
            for log_file in log_files[:3]:  # Show 3 most recent
                log_path = os.path.join(log_dir, log_file)
                print(f"\n   📄 {log_file}:")
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        recent_lines = [line for line in lines[-10:] if 'Epoch' in line or 'Loss' in line]
                        for line in recent_lines[-3:]:  # Show last 3 relevant lines
                            print(f"      {line.strip()}")
                except Exception as e:
                    print(f"      ❌ Error reading log: {e}")
    
    def cleanup_sessions(self):
        """Clean up dead sessions"""
        print("🧹 Cleaning up dead sessions...")
        
        # Clean tmux sessions
        try:
            result = subprocess.run(["tmux", "list-sessions"], capture_output=True)
            if result.returncode != 0:
                print("   No tmux sessions to clean")
        except FileNotFoundError:
            pass
        
        # Clean screen sessions
        try:
            result = subprocess.run(["screen", "-wipe"], capture_output=True)
            print("   Screen sessions cleaned")
        except FileNotFoundError:
            pass

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Session Manager")
    parser.add_argument("action", choices=["status", "cleanup"], 
                       help="Action to perform")
    
    args = parser.parse_args()
    
    manager = SessionManager()
    
    if args.action == "status":
        manager.show_status()
    elif args.action == "cleanup":
        manager.cleanup_sessions()

if __name__ == "__main__":
    main()
