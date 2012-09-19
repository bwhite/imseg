import sys
import subprocess


def main():
    try:
        num_procs = int(sys.argv[1])
    except:
        print('Usage: python driver_multi.py <num_procs> [driver_args...]')
        sys.exit(1)
    procs = []
    for x in range(num_procs):
        procs.append(subprocess.Popen(['python', 'driver.py'] + sys.argv[2:]))
    for p in procs:
        p.wait()
main()
