import time
from ray.job_submission import JobSubmissionClient, JobStatus

class RayJobManager:
    def __init__(self, ray_address: str):
        self.ray_address = ray_address
        self.client = JobSubmissionClient(ray_address)

    def submit_job(self, entrypoint: str, runtime_env=None) -> str:
        """
        Submit a Ray job and return the job ID.
        """
        if runtime_env is None:
            runtime_env = {}

        job_id = self.client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            entrypoint_num_gpus=2,
        )
        print(f"Submitted Ray job: {job_id}")
        return job_id

    def wait_until_done(self, job_id: str, timeout_seconds: int = 3600):
        """
        Wait for a job to finish and print logs.
        """
        start = time.time()
        while time.time() - start < timeout_seconds:
            status = self.client.get_job_status(job_id)
            print(f"Job status: {status}")
            if status in {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}:
                break
            time.sleep(10)
        else:
            print("⚠️ Timeout waiting for job.")
            return

        print("\n--- Job Logs ---")
        logs = self.client.get_job_logs(job_id)
        print(logs)

        if status == JobStatus.SUCCEEDED:
            print("Job completed successfully!")
        elif status == JobStatus.FAILED:
            print("Job failed.")
        else:
            print("Job stopped manually.")

