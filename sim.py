import random,argparse,sys,subprocess,os

i = 0
while True:
    L = 6
    cmd_encoder = "python", "generate_mdp.py", "--K", str(0.01), "--len_action", str(L), "--seed", str(i)
    f = open('mdp_file','w')
    subprocess.call(cmd_encoder,stdout=f)
    f.close()
    p = False
    pp = False
    fix_pol = ""
    for length in range(1, L):
        cmd_encoder = "python", "planner.py","--mdp", "mdp_file", "--optimal", "--len_action", str(length)
        output = subprocess.check_output(cmd_encoder,universal_newlines=True)
        policy = output.split("\n")[0]
        a0 = policy.split(" ")[0]
        a1 = policy.split(" ")[1]
        if (not p) and len(a0)<length and len(a1)<length:
            p = True
            fix_pol = policy
            # print(i)
        if p:
            if policy != fix_pol:
                pp = True
                print(i, policy, fix_pol, length)
                exit(0)
    if pp:
        print("")
    os.remove("mdp_file")
    i+=1