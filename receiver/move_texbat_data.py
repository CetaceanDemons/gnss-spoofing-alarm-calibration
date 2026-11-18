import os
import shutil
import sys

def move_data(scenario_name):
    # dump folder of the receiver (virtual machine)
    source_dir = os.path.expanduser("~/gnss-sdr/processed_Mao_TEXBAT/temp_dump")
    # shared folder of the host
    target_base = "/mnt/hgfs/Database/TEXBAT/processed_Mao"
    target_dir = os.path.join(target_base, scenario_name)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files_moved = 0
    for filename in os.listdir(source_dir):
        src_file = os.path.join(source_dir, filename)
        
        # keep only the .dat files
        if filename.endswith(".dat"):
            dst_file = os.path.join(target_dir, filename)
            shutil.move(src_file, dst_file)
            files_moved += 1
        else:
            # delete the other files (empty .mat placeholders)
            if os.path.isfile(src_file):
                os.remove(src_file)

    print(f"{scenario_name}: {files_moved} .dat files moved to {target_dir}; other files deleted")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 move_texbat_data.py <scenario>")
        sys.exit(1)
    
    scenario = sys.argv[1]
    move_data(scenario)
