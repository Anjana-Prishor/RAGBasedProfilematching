"""Generate a diverse, reproducible resume dataset for the project."""

from pathlib import Path


PROFILES = [
    ("Aisha Rahman", "Machine Learning Engineer", 5, "MSc Computer Science", ["Python", "PyTorch", "TensorFlow", "NLP", "AWS"], "recommendation systems and model deployment"),
    ("Ben Carter", "Data Engineer", 7, "BSc Software Engineering", ["Python", "SQL", "Spark", "Airflow", "Snowflake"], "batch pipelines and warehouse modeling"),
    ("Chloe Nguyen", "Frontend Engineer", 4, "BFA Interaction Design", ["JavaScript", "TypeScript", "React", "Vue", "CSS"], "accessible dashboards and design systems"),
    ("Daniel Okafor", "Backend Engineer", 6, "BEng Computer Engineering", ["Java", "Spring", "PostgreSQL", "Kafka", "Docker"], "high-throughput APIs and event-driven services"),
    ("Elena Rossi", "Cloud Engineer", 8, "MSc Information Systems", ["AWS", "Terraform", "Kubernetes", "Linux", "Python"], "multi-account cloud platforms and infrastructure automation"),
    ("Farah Malik", "Product Analyst", 3, "BSc Statistics", ["SQL", "Python", "Tableau", "Excel", "A/B Testing"], "funnel analysis and product experimentation"),
    ("George Wilson", "Security Engineer", 9, "BSc Cybersecurity", ["Python", "Linux", "AWS", "SIEM", "Threat Modeling"], "cloud security monitoring and incident response"),
    ("Hana Suzuki", "QA Automation Engineer", 5, "BSc Information Technology", ["Python", "Selenium", "Cypress", "API Testing", "Jenkins"], "reliable web and API test automation"),
    ("Ibrahim Diallo", "DevOps Engineer", 6, "BSc Computer Science", ["Docker", "Kubernetes", "Jenkins", "Terraform", "AWS"], "continuous delivery and observability platforms"),
    ("Julia Meyer", "Mobile Engineer", 4, "BSc Computer Science", ["Kotlin", "Android", "Java", "Firebase", "REST API"], "offline-first Android applications"),
    ("Kai Thompson", "Data Scientist", 5, "MSc Data Science", ["Python", "Pandas", "Scikit-Learn", "SQL", "Machine Learning"], "forecasting and customer segmentation"),
    ("Lina Petrova", "Full Stack Developer", 7, "BSc Computer Science", ["React", "Node", "TypeScript", "PostgreSQL", "GraphQL"], "customer-facing SaaS products"),
    ("Marcus Lee", "Solutions Architect", 11, "MSc Computer Engineering", ["AWS", "Azure", "Microservices", "Docker", "Java"], "scalable enterprise integration architectures"),
    ("Nora Singh", "NLP Researcher", 4, "PhD Computational Linguistics", ["Python", "PyTorch", "NLP", "Transformers", "Research"], "information extraction and language models"),
    ("Owen Brooks", "Database Administrator", 10, "BSc Information Systems", ["PostgreSQL", "MySQL", "Oracle", "Linux", "Backup"], "database reliability and performance tuning"),
    ("Priya Shah", "Scrum Product Manager", 8, "MBA Technology Management", ["Roadmaps", "Agile", "Analytics", "Jira", "Stakeholder Management"], "data-informed platform roadmaps"),
    ("Quinn Adams", "Computer Vision Engineer", 6, "MSc Robotics", ["Python", "OpenCV", "PyTorch", "Computer Vision", "C++"], "industrial image inspection systems"),
    ("Ravi Patel", "Site Reliability Engineer", 7, "BEng Software Engineering", ["Kubernetes", "Prometheus", "Grafana", "Go", "Linux"], "reliable distributed services and observability"),
    ("Sofia Garcia", "UX Researcher", 5, "MSc Human Computer Interaction", ["User Research", "Figma", "Usability Testing", "Analytics", "Prototyping"], "qualitative research for workflow tools"),
    ("Theo Martin", "Java Developer", 6, "BSc Computer Science", ["Java", "Spring", "Kafka", "SQL", "Microservices"], "financial transaction services"),
    ("Uma Nair", "Analytics Engineer", 4, "BSc Mathematics", ["SQL", "dbt", "Snowflake", "Python", "Looker"], "trusted metrics layers and reporting"),
    ("Victor Chen", "React Developer", 3, "BSc Web Development", ["React", "JavaScript", "TypeScript", "Jest", "Git"], "responsive component libraries"),
    ("Wendy Brown", "Platform Engineer", 9, "MSc Cloud Computing", ["Go", "Kubernetes", "Terraform", "AWS", "Linux"], "internal developer platforms"),
    ("Xavier Laurent", "Applied Scientist", 8, "PhD Machine Learning", ["Python", "TensorFlow", "Deep Learning", "NLP", "AWS"], "ranking models and applied experimentation"),
    ("Yara Hassan", "ETL Developer", 5, "BSc Data Engineering", ["SQL", "Python", "Airflow", "Spark", "Azure"], "batch ingestion and data quality"),
    ("Zachary King", "API Engineer", 4, "BSc Software Engineering", ["Node", "Express", "REST API", "MongoDB", "Docker"], "secure REST and event APIs"),
    ("Amara Williams", "Cloud Security Analyst", 6, "BSc Cybersecurity", ["AWS", "Azure", "IAM", "Python", "SIEM"], "identity controls and cloud compliance"),
    ("Bruno Silva", "Python Developer", 2, "BSc Computer Science", ["Python", "Django", "PostgreSQL", "REST API", "Git"], "maintainable web services"),
    ("Celeste Kim", "BI Developer", 5, "BSc Business Analytics", ["SQL", "Power BI", "DAX", "Azure", "ETL"], "executive reporting and operational dashboards"),
    ("Diego Alvarez", "ML Platform Engineer", 7, "MEng Artificial Intelligence", ["Python", "Kubernetes", "MLflow", "Docker", "AWS"], "reproducible machine learning operations"),
]


def write_resumes(output_dir: str = "sample_resumes") -> int:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for index, (name, title, years, education, skills, focus) in enumerate(PROFILES, 1):
        filename = directory / f"generated_{index:02d}_{name.lower().replace(' ', '_')}.txt"
        text = (
            f"{name}\n"
            f"{title} | {years} years of experience\n\n"
            "SUMMARY\n"
            f"{title} focused on {focus}. Experienced in collaborating with product and engineering teams.\n\n"
            "EXPERIENCE\n"
            f"{title}, Innovation Labs (2018-Present)\n"
            f"- Delivered production systems for {focus}.\n"
            f"- Improved reliability, quality, or delivery speed through automation and measurement.\n"
            f"- Technical focus: {', '.join(skills)}.\n\n"
            "EDUCATION\n"
            f"{education}\n\n"
            "SKILLS\n"
            f"{', '.join(skills)}\n"
        )
        filename.write_text(text, encoding="utf-8")
    return len(PROFILES)


if __name__ == "__main__":
    count = write_resumes()
    print(f"Generated {count} additional resumes in sample_resumes/")
