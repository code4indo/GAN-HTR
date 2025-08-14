#!/usr/bin/env python3
"""
Resource monitoring script untuk training GAN-HTR
Memantau CPU, RAM, GPU, Storage secara real-time
"""

import os
import time
import psutil
import subprocess
import json
from datetime import datetime
import threading
import signal
import sys

class ResourceMonitor:
    def __init__(self, log_file="resource_monitor.log"):
        self.log_file = log_file
        self.monitoring = False
        self.data_points = []
        
    def get_cpu_info(self):
        """Get CPU usage information"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        
        return {
            'cpu_percent_total': sum(cpu_percent) / len(cpu_percent),
            'cpu_percent_per_core': cpu_percent,
            'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
            'cpu_freq_max': cpu_freq.max if cpu_freq else 0,
            'cpu_count': cpu_count
        }
    
    def get_memory_info(self):
        """Get memory usage information"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'memory_total_gb': memory.total / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_percent': swap.percent
        }
    
    def get_gpu_info(self):
        """Get GPU usage information"""
        try:
            # Get GPU info using nvidia-smi
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            gpus = []
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 9:
                            gpus.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_total_mb': int(parts[2]),
                                'memory_used_mb': int(parts[3]),
                                'memory_free_mb': int(parts[4]),
                                'gpu_utilization_percent': int(parts[5]),
                                'temperature_c': int(parts[6]),
                                'power_draw_w': float(parts[7]) if parts[7] != 'N/A' else 0,
                                'power_limit_w': float(parts[8]) if parts[8] != 'N/A' else 0
                            })
            
            return gpus
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
            print(f"Error getting GPU info: {e}")
            return []
    
    def get_storage_info(self):
        """Get storage usage information"""
        storage_info = {}
        
        # Get disk usage for main partition
        disk_usage = psutil.disk_usage('/')
        storage_info['/'] = {
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3),
            'free_gb': disk_usage.free / (1024**3),
            'percent': (disk_usage.used / disk_usage.total) * 100
        }
        
        # Get disk I/O stats
        try:
            disk_io = psutil.disk_io_counters()
            storage_info['io'] = {
                'read_bytes_per_sec': disk_io.read_bytes,
                'write_bytes_per_sec': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            }
        except:
            storage_info['io'] = {}
            
        return storage_info
    
    def get_process_info(self, process_name="python"):
        """Get process-specific information"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
            try:
                if process_name.lower() in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'train_gan' in cmdline:
                        processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / (1024**2) if proc.info['memory_info'] else 0,
                            'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return processes
    
    def collect_data(self):
        """Collect all monitoring data"""
        timestamp = datetime.now().isoformat()
        
        data = {
            'timestamp': timestamp,
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'gpu': self.get_gpu_info(),
            'storage': self.get_storage_info(),
            'processes': self.get_process_info()
        }
        
        return data
    
    def print_summary(self, data):
        """Print formatted summary"""
        print(f"\n{'='*80}")
        print(f"RESOURCE MONITOR - {data['timestamp'][:19]}")
        print(f"{'='*80}")
        
        # CPU Info
        cpu = data['cpu']
        print(f"🔥 CPU (AMD Threadripper PRO 3955WX):")
        print(f"   Usage: {cpu['cpu_percent_total']:.1f}% | Freq: {cpu['cpu_freq_current']:.0f}MHz | Cores: {cpu['cpu_count']}")
        
        # Memory Info
        mem = data['memory']
        print(f"💾 RAM (128GB Total):")
        print(f"   Used: {mem['memory_used_gb']:.1f}GB ({mem['memory_percent']:.1f}%) | Available: {mem['memory_available_gb']:.1f}GB")
        
        # GPU Info
        print(f"🚀 GPU (Dual RTX A4000):")
        for i, gpu in enumerate(data['gpu']):
            util = gpu['gpu_utilization_percent']
            mem_used = gpu['memory_used_mb'] / 1024
            mem_total = gpu['memory_total_mb'] / 1024
            mem_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
            temp = gpu['temperature_c']
            power = gpu['power_draw_w']
            
            print(f"   GPU{i}: {util:2d}% util | {mem_used:.1f}/{mem_total:.1f}GB ({mem_percent:.1f}%) | {temp}°C | {power:.0f}W")
        
        # Storage Info
        storage = data['storage']['/']
        print(f"💿 Storage (NVMe SSD):")
        print(f"   Used: {storage['used_gb']:.1f}GB ({storage['percent']:.1f}%) | Free: {storage['free_gb']:.1f}GB")
        
        # Training Processes
        if data['processes']:
            print(f"🏃 Training Processes:")
            for proc in data['processes']:
                print(f"   PID {proc['pid']}: {proc['cpu_percent']:.1f}% CPU | {proc['memory_mb']:.0f}MB RAM")
    
    def save_to_log(self, data):
        """Save data to log file"""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def monitor_loop(self, interval=5):
        """Main monitoring loop"""
        print(f"🔍 Starting resource monitoring (interval: {interval}s)")
        print(f"📝 Logging to: {self.log_file}")
        print("Press Ctrl+C to stop")
        
        try:
            while self.monitoring:
                data = self.collect_data()
                self.data_points.append(data)
                self.print_summary(data)
                self.save_to_log(data)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")
    
    def start_monitoring(self, interval=5):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        return self.monitor_thread
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
    
    def generate_report(self):
        """Generate performance report"""
        if not self.data_points:
            print("No data points collected yet")
            return
        
        print(f"\n{'='*80}")
        print("PERFORMANCE REPORT")
        print(f"{'='*80}")
        
        # Calculate averages
        cpu_avg = sum(d['cpu']['cpu_percent_total'] for d in self.data_points) / len(self.data_points)
        mem_avg = sum(d['memory']['memory_percent'] for d in self.data_points) / len(self.data_points)
        
        if self.data_points[0]['gpu']:
            gpu0_util_avg = sum(d['gpu'][0]['gpu_utilization_percent'] for d in self.data_points if d['gpu']) / len([d for d in self.data_points if d['gpu']])
            gpu1_util_avg = sum(d['gpu'][1]['gpu_utilization_percent'] for d in self.data_points if len(d['gpu']) > 1) / len([d for d in self.data_points if len(d['gpu']) > 1])
        else:
            gpu0_util_avg = gpu1_util_avg = 0
        
        print(f"Data Points: {len(self.data_points)}")
        print(f"Average CPU Usage: {cpu_avg:.1f}%")
        print(f"Average RAM Usage: {mem_avg:.1f}%")
        print(f"Average GPU0 Usage: {gpu0_util_avg:.1f}%")
        print(f"Average GPU1 Usage: {gpu1_util_avg:.1f}%")
        
        # Resource utilization assessment
        print(f"\n📊 UTILIZATION ASSESSMENT:")
        
        if cpu_avg < 50:
            print(f"⚠️  CPU underutilized ({cpu_avg:.1f}%) - Consider increasing batch size or workers")
        elif cpu_avg > 90:
            print(f"🔥 CPU highly utilized ({cpu_avg:.1f}%) - Excellent!")
        else:
            print(f"✅ CPU well utilized ({cpu_avg:.1f}%)")
        
        if mem_avg < 30:
            print(f"⚠️  RAM underutilized ({mem_avg:.1f}%) - Consider increasing batch size")
        elif mem_avg > 85:
            print(f"⚠️  RAM highly utilized ({mem_avg:.1f}%) - Monitor for OOM")
        else:
            print(f"✅ RAM well utilized ({mem_avg:.1f}%)")
        
        if gpu0_util_avg < 50:
            print(f"⚠️  GPU0 underutilized ({gpu0_util_avg:.1f}%) - Consider increasing batch size")
        else:
            print(f"✅ GPU0 well utilized ({gpu0_util_avg:.1f}%)")
        
        if gpu1_util_avg < 50:
            print(f"⚠️  GPU1 underutilized ({gpu1_util_avg:.1f}%) - Check multi-GPU setup")
        else:
            print(f"✅ GPU1 well utilized ({gpu1_util_avg:.1f}%)")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n🛑 Received interrupt signal, stopping monitor...')
    if 'monitor' in globals():
        monitor.stop_monitoring()
    sys.exit(0)

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Resource Monitor for GAN-HTR Training')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds')
    parser.add_argument('--log', type=str, default='resource_monitor.log', help='Log file name')
    parser.add_argument('--report', action='store_true', help='Generate report from existing log')
    
    args = parser.parse_args()
    
    global monitor
    monitor = ResourceMonitor(log_file=args.log)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    if args.report:
        # Load existing log and generate report
        try:
            with open(args.log, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    monitor.data_points.append(data)
            monitor.generate_report()
        except FileNotFoundError:
            print(f"Log file {args.log} not found")
        except Exception as e:
            print(f"Error reading log file: {e}")
    else:
        # Start real-time monitoring
        print("🚀 GAN-HTR Resource Monitor")
        print("="*50)
        print("Hardware Summary:")
        print("  CPU: AMD Threadripper PRO 3955WX (32 threads)")
        print("  RAM: 128GB DDR4")
        print("  GPU: 2x NVIDIA RTX A4000 (16GB each)")
        print("  Storage: PNY CS3040 2TB NVMe SSD")
        print("="*50)
        
        monitor.start_monitoring(interval=args.interval)
        
        try:
            # Keep main thread alive
            while monitor.monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            monitor.stop_monitoring()
            monitor.generate_report()

if __name__ == "__main__":
    main()
