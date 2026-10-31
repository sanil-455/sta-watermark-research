import os, re, sys, time
path = sys.argv[1] if len(sys.argv) > 1 else "controls.log"
log = open(path).read()
if "finished:" in log:
    print(log.strip().splitlines()[-1])
    raise SystemExit
done = re.findall(r"^\[(\d+)/(\d+)\] prompt \d+ \|.*about (\d+) min left", log, re.M)
short = len(re.findall(r"^prompt \d+: model stopped", log, re.M))
if not done:
    raise SystemExit("no usable sample in %s yet: wait a few minutes (model loading takes 1 to 2)" % path)
k, target, eta = (int(x) for x in done[-1])
rate = len(done) / (len(done) + short)
left = eta / rate
age = (time.time() - os.path.getmtime(path)) / 60
print("usable: %d of %d | attempts so far %d (%.0f%% usable)" % (k, target, len(done) + short, 100 * rate))
print("log's own estimate: %d min (assumes every attempt is usable)" % eta)
print("realistic time left: about %.0f min, finishing around %s" % (left, time.strftime("%H:%M", time.localtime(time.time() + 60 * left))))
if age > 3:
    print("WARNING: no new line for %.0f min. Check the process: pgrep -af generate" % age)
