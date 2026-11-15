"""A stand-in for the gcloud CLI, used by tests and local UI checks (never touches real credentials).

Install it first on PATH as `gcloud` (see tests/test_auth_flow.py::fake_gcloud). Behaviour via environment:
  FAKE_GCLOUD_LOG       file that receives one JSON line per call ({"argv": [...], "code": "..."})
  FAKE_GCLOUD_MODE      "ok" (default), "no_url" (prints nothing, waits), "fail_start" (exits 1 at once)
  FAKE_GCLOUD_GOOD_CODE the code that "succeeds" (default "good-code")
  FAKE_GCLOUD_PROJECTS  JSON for `gcloud projects list`; unset -> the command fails
"""

import json
import os
import sys
import time

FAKE_URL = ("https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=fake-client.apps"
            ".googleusercontent.com&redirect_uri=https%3A%2F%2Fsdk.cloud.google.com%2Fapplicationdefaultauthcode.html"
            "&scope=openid+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform&state=fake-state"
            "&prompt=consent&token_usage=remote&access_type=offline&code_challenge=abc&code_challenge_method=S256")


def log(entry: dict) -> None:
    path = os.environ.get("FAKE_GCLOUD_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def main(argv: list[str]) -> int:
    log({"argv": argv, "disable_prompts": os.environ.get("CLOUDSDK_CORE_DISABLE_PROMPTS")})
    if argv[:1] == ["auth"] and "login" in argv:
        mode = os.environ.get("FAKE_GCLOUD_MODE", "ok")
        if mode == "fail_start":
            sys.stderr.write("ERROR: (gcloud.auth.login) something went wrong before the link\n")
            return 1
        if mode == "no_url":
            time.sleep(60)
            return 1
        sys.stderr.write("Go to the following link in your browser, and complete the sign-in prompts:\n\n"
                         f"    {FAKE_URL}\n\n")
        sys.stderr.flush()
        code = input("Once finished, enter the verification code provided in your browser: ").strip()
        log({"argv": argv, "code": code})
        if code == os.environ.get("FAKE_GCLOUD_GOOD_CODE", "good-code"):
            if "application-default" in argv:
                sys.stderr.write("\nCredentials saved to file: [/tmp/fake-adc.json]\n\nThese credentials will be "
                                 "used by any library that requests Application Default Credentials (ADC).\n")
            else:
                sys.stderr.write("\nYou are now logged in as [you@example.com].\n")
            return 0
        sys.stderr.write("ERROR: There was a problem with web authentication. Try running again with "
                         "--no-launch-browser.\nERROR: (gcloud.auth.application-default.login) (invalid_grant) "
                         "Malformed auth code.\n")
        return 1
    if argv[:3] == ["config", "get-value", "project"]:
        print(os.environ.get("FAKE_GCLOUD_PROJECT", "my-gcp-project"))
        return 0
    if argv[:3] == ["config", "get-value", "account"]:
        print("you@example.com")
        return 0
    if argv[:2] == ["config", "set"]:
        sys.stderr.write("Updated property [core/project].\n")
        return 0
    if argv[:2] == ["auth", "print-access-token"]:
        sys.stderr.write("ERROR: (gcloud.auth.print-access-token) fake gcloud has no token\n")
        return 1
    if argv[:2] == ["projects", "list"]:
        data = os.environ.get("FAKE_GCLOUD_PROJECTS")
        if data is None:
            sys.stderr.write("ERROR: (gcloud.projects.list) fake failure\n")
            return 1
        print(data)
        return 0
    sys.stderr.write(f"fake gcloud: unsupported command {argv}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
