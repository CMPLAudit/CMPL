import pandas as pd
import random

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def split_free_slots(free_slots, num_people = 100):
    remote_free_slots = {i:{day:[] for day in days_of_week} for i in range(num_people)}
    inperson_free_slots = {i:{day:[] for day in days_of_week} for i in range(num_people)}
    for i in range(num_people):
        remote_free_slots[i]
        for day in days_of_week:
            shuffled_free_slots = random.sample(free_slots[i][day], len(free_slots[i][day]))
            remote_free_slots[i][day], inperson_free_slots[i][day] = shuffled_free_slots[0], shuffled_free_slots[1]

    return remote_free_slots, inperson_free_slots

if __name__ == "__main__":
    # Define people and days of the week
    people = [f"Person_{i}" for i in range(0, 200)]  # 200 people for two partitions

    activities = ["Work", "Sleep", "Exercise", "Meeting", "Socializing", "Shopping", "Study", "Relax", "Chores"]

    interviewer_activities = ["Work", "Sleep", "Relax", "Chores"]

    # Enhanced details for activities
    hospitals = ["City Hospital", "General Hospital", "Downtown Clinic"]
    therapists = ["Dr. Smith", "Dr. Johnson", "Dr. Patel"]
    confidential_entities = ["Client A", "Corporate X", "Government Agency"]
    financial_advisors = ["Advisor Brown", "Advisor Green", "Advisor White"]

    companies = ["TechCorp", "BizSolutions", "DataPros"]
    interview_companies = ["Monarch", "Oscorp", "Lumon Industries"]
    gyms = ["FitLife Gym", "PowerZone", "Flex Fitness"]
    teams = ["Marketing Team", "Development Team", "HR Team"]
    friends = ["Alice", "Bob", "Charlie"]
    stores = ["Mall Center", "SuperMart", "GroceryPlus"]
    subjects = ["Math", "Physics", "Economics"]
    relax_locations = ["Park", "Home", "Beach"]

    # Update schedule to include 7-9 hours of sleep and free slots
    enhanced_schedule_with_sleep_and_free = []
    sleep_hours_range = range(7, 10)

    # Generate free slots for partition 1 and partition 2
    partition_1_free_slots = [{day:random.sample(range(9, 17), 2) for day in days_of_week} for _ in range(100)]
    partition_2_free_slots = [{day:random.sample(range(9, 17), 2) for day in days_of_week} for _ in range(100)]

    partition_1_free_slots_remote, partition_1_free_slots_inperson = split_free_slots(partition_1_free_slots)
    partition_2_free_slots_remote, partition_2_free_slots_inperson = split_free_slots(partition_2_free_slots)

    # Ensure at least one matching free slot between partitions on 2 days
    for i in range(100):
        for day in random.sample(days_of_week,2):
            common_free_slot = partition_1_free_slots_remote[i][day]
            partition_2_free_slots_remote[i][day] = common_free_slot

    for i in range(100):
        for day in random.sample(days_of_week,2):
            common_free_slot = partition_1_free_slots_inperson[i][day]
            partition_2_free_slots_inperson[i][day] = common_free_slot

    for i, person in enumerate(people):
        partition = 1 if i < 100 else 2
        free_slots_remote = partition_1_free_slots_remote[i] if partition == 1 else partition_2_free_slots_remote[i - 100]
        free_slots_inperson = partition_1_free_slots_inperson[i] if partition == 1 else partition_2_free_slots_inperson[i - 100]
        company_name = random.choice(companies)
        gym_name = random.choice(gyms)
        for day in days_of_week:
            day_free_slots_remote, day_free_slots_inperson = [free_slots_remote[day]], [free_slots_inperson[day]]
            sleep_hours = random.choice(sleep_hours_range)  # 7 to 9 hours of sleep
            sleep_start = random.randint(21, 23)  # Sleep starts between 9 PM and 11 PM
            
            for hour in range(24):
                hour_str = f"{hour:02}:00"
                
                # Assign sleep hours
                if sleep_start <= hour or hour < (sleep_start + sleep_hours - 24):
                    activity = "Sleep"
                # Assign free slots
                elif hour in day_free_slots_remote:
                    activity = "Free"
                elif hour in day_free_slots_inperson:
                    activity = "Free"
                # Assign other activities
                else:
                    if partition == 1:
                        if random.random() < 0.2:  # 10% chance of a sensitive activity
                            if random.choice([True, False]):
                                if random.choice([True, False]):
                                    activity = f"Medical Appointment at {random.choice(hospitals)}"
                                else:
                                    activity = f"Confidential Meeting with {random.choice(confidential_entities)}"
                            else:
                                if random.choice([True, False]):
                                    activity = f"Job interview with {random.choice(interview_companies)}"
                                else:
                                    activity = f"Legal consultation with {random.choice(financial_advisors)}"
                        else:
                            activity = random.choice([
                                f"Work at {company_name}",
                                f"Exercise at {gym_name}",
                                f"Meeting with {random.choice(teams)}",
                                f"Socializing with {random.choice(friends)}",
                                f"Shopping at {random.choice(stores)}",
                                f"Study for {random.choice(subjects)}",
                                f"Relax at {random.choice(relax_locations)}",
                                "House Chores"
                            ])
                    else:
                        activity = "Other Activities (Not Free)"
                
                enhanced_schedule_with_sleep_and_free.append([person, day, hour_str, activity, partition])

    # Create DataFrame with sleep and free slots
    enhanced_schedule_with_sleep_and_free_df = pd.DataFrame(
        enhanced_schedule_with_sleep_and_free, columns=["Person", "Day", "Hour", "Activity", "Partition"]
    )

    # Save to CSV
    enhanced_schedule_with_sleep_and_free_csv_path = '/path/to/schedule.csv'
    enhanced_schedule_with_sleep_and_free_df.to_csv(enhanced_schedule_with_sleep_and_free_csv_path, index=False)

    # Verification script
    partition_1_people = enhanced_schedule_with_sleep_and_free_df[enhanced_schedule_with_sleep_and_free_df["Partition"] == 1]
    partition_2_people = enhanced_schedule_with_sleep_and_free_df[enhanced_schedule_with_sleep_and_free_df["Partition"] == 2]

    def get_free_slots(df):
        free_slots = {}
        for person in df["Person"].unique():
            person_df = df[df["Person"] == person]
            free_slots[person] = person_df[person_df["Activity"] == "Free"]["Hour"].tolist()
        return free_slots

    partition_1_free_slots = get_free_slots(partition_1_people)
    partition_2_free_slots = get_free_slots(partition_2_people)

    def verify_matching_free_slots(partition_1_free_slots, partition_2_free_slots):
        for person_1, free_slots_1 in partition_1_free_slots.items():
            match_found = False
            for person_2, free_slots_2 in partition_2_free_slots.items():
                if set(free_slots_1).intersection(set(free_slots_2)):
                    match_found = True
                    print(f'{person_1} and {person_2} have matching free slots.')
                    break
            if not match_found:
                return False
        return True

    partition_1_to_2_verified_remote = verify_matching_free_slots(partition_1_free_slots, partition_2_free_slots)
    partition_2_to_1_verified_remote = verify_matching_free_slots(partition_2_free_slots, partition_1_free_slots)

    print(f"Partition 1 to Partition 2 verification: {partition_1_to_2_verified_remote}")
    print(f"Partition 2 to Partition 1 verification: {partition_2_to_1_verified_remote}")