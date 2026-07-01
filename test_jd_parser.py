from src.jd_parser import parse_job_description

jd = parse_job_description("data/jobs/job_description.docx")

print("\nExtracted Job Profile:")
print(jd)