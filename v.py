import json
import pandas as pd
from django.shortcuts import render
from .models import TripDataUpload

def daily_report_view(request):
    trip_file = TripDataUpload.objects.filter(file_type='trip').order_by('-uploaded_at').first()
    testdata_file = TripDataUpload.objects.filter(file_type='testdata').order_by('-uploaded_at').first()

    if not trip_file or not testdata_file:
        return render(request, 'tripdata/daily_report.html', {
            'error': 'Please upload both Trip and TestData files first.'
        })

    trip_cols = [
        'customer_phone_no', 'fare', 'trip_status', 'cancelled_by',
        'created_at', 'car_type', 'no_of_bids',
        'customer_created_at', 'customer_no_of_trips', 'pickup_div'
    ]
    testdata_cols = ['Testing Number']

    # read - use cols safely
    trip_df = pd.read_excel(trip_file.file.path, usecols=lambda x: x in trip_cols)
    testdata_df = pd.read_excel(testdata_file.file.path, usecols=lambda x: x in testdata_cols)

    # Clean phone numbers
    trip_df['clean_phone'] = trip_df['customer_phone_no'].astype(str).str.strip().str.replace(' ', '').str.lstrip('0')
    testdata_df['clean_test_number'] = testdata_df['Testing Number'].astype(str).str.strip().str.replace(' ', '').str.lstrip('0')
    trip_df = trip_df[~trip_df['clean_phone'].isin(testdata_df['clean_test_number'])]

    # Dates
    trip_df['created_at'] = pd.to_datetime(trip_df['created_at'], errors='coerce')
    trip_df = trip_df.dropna(subset=['created_at'])
    trip_df['date'] = trip_df['created_at'].dt.date
    trip_df['weekday'] = trip_df['created_at'].dt.strftime('%A')

    # Car type cleanup
    trip_df['car_type'] = trip_df['car_type'].astype(str).fillna('Unknown').replace('nan', 'Unknown')
    car_types = sorted(trip_df['car_type'].unique())
    car_types_with_all = ['All'] + car_types

    # Pickup division cleanup
    trip_df['pickup_div'] = trip_df['pickup_div'].astype(str).fillna('Unknown').replace('nan', 'Unknown')
    pickup_divs = sorted(trip_df['pickup_div'].unique())
    pickup_divs_with_all = ['All'] + pickup_divs

    # Zero bids
    trip_df['no_of_bids'] = pd.to_numeric(trip_df['no_of_bids'], errors='coerce')
    trip_df['is_zero_bid'] = trip_df['no_of_bids'].isna() | (trip_df['no_of_bids'] == 0)

    # 👉 NEW: Add bid range categories
    trip_df['bid_range'] = pd.cut(
        trip_df['no_of_bids'],
        bins=[-1, 0, 5, 10, float('inf')],
        labels=['Zero Bid', '2-5 Bid', '6-10 Bid', '10+ Bid']
    )

    # Customer age classification
    trip_df['customer_created_at'] = pd.to_datetime(trip_df.get('customer_created_at'), errors='coerce')
    trip_df['days_since_signup'] = (trip_df['created_at'] - trip_df['customer_created_at']).dt.days
    trip_df['customer_age_type'] = trip_df['days_since_signup'].apply(lambda x: 'New' if pd.notna(x) and x < 30 else 'Old')

    # Repeat / non-repeat
    trip_df['repeat_type'] = trip_df['customer_no_of_trips'].apply(
        lambda x: 'Repeat' if pd.notna(x) and str(x).strip() not in ['', '0', 'nan'] else 'Non-Repeat'
    )

    # Group by date
    grouped = trip_df.groupby('date', sort=True)

    daily_rows = []
    totals_acc = {k: 0 for k in [
        'total', 'confirmed', 'non_confirmed', 'trip_still_confirmed',
        'completed', 'cancelled', 'cancelled_by_customer', 'cancelled_by_driver', 'zero_bid',
        # 👉 NEW
        'bid_2_5', 'bid_6_10', 'bid_10_plus'
    ]}

    def pct(val, parent):
        return f"{(val / parent * 100):.0f}%" if parent else "0%"

    # Top-level arrays (series)
    dates = []
    totalTrips = []
    confirmedTrips = []
    nonConfirmedTrips = []
    completedTrips = []
    cancelledTrips = []
    zeroBidTrips = []

    # 👉 NEW arrays
    bid2_5Trips = []
    bid6_10Trips = []
    bid10PlusTrips = []

    # arrays for age & repeat series (overall All car type)
    newCustomerTrips = []
    oldCustomerTrips = []
    repeatTrips = []
    nonRepeatTrips = []

    # Prepare breakdown nested structure:
    metrics = [
        'total', 'confirmed', 'non_confirmed', 'completed', 'cancelled_from_confirmed', 'zero_bid',
        # 👉 NEW
        'bid_2_5', 'bid_6_10', 'bid_10_plus'
    ]
    age_keys = ['All', 'New', 'Old']
    repeat_keys = ['All', 'Repeat', 'Non-Repeat']

    breakdown = {}
    for metric in metrics:
        breakdown[metric] = {}
        for ct in car_types_with_all:
            breakdown[metric][ct] = {}
            for ak in age_keys:
                breakdown[metric][ct][ak] = {}
                for rk in repeat_keys:
                    breakdown[metric][ct][ak][rk] = {}
                    for pdv in pickup_divs_with_all:
                        breakdown[metric][ct][ak][rk][pdv] = []

    # iterate days and compute numbers
    for day, group in grouped:
        dates.append(f"{group['weekday'].iloc[0]}, {day.strftime('%d %b %Y')}")
        total = len(group)
        confirmed = int(group['fare'].notna().sum())
        non_confirmed = int(total - confirmed)
        completed = int(((group['fare'].notna()) & (~group['trip_status'].isin(['Trip Confirmed', 'Cancelled']))).sum())
        cancelled_from_confirmed = int(((group['fare'].notna()) & (group['trip_status'] == 'Cancelled')).sum())
        zero_bid = int(group['is_zero_bid'].sum())

        # 👉 NEW bid counts
        bid_2_5 = int(((group['no_of_bids'] >= 2) & (group['no_of_bids'] <= 5)).sum())
        bid_6_10 = int(((group['no_of_bids'] >= 6) & (group['no_of_bids'] <= 10)).sum())
        bid_10_plus = int((group['no_of_bids'] > 10).sum())

        totalTrips.append(int(total))
        confirmedTrips.append(confirmed)
        nonConfirmedTrips.append(non_confirmed)
        completedTrips.append(completed)
        cancelledTrips.append(cancelled_from_confirmed)
        zeroBidTrips.append(zero_bid)

        # 👉 NEW append to arrays
        bid2_5Trips.append(bid_2_5)
        bid6_10Trips.append(bid_6_10)
        bid10PlusTrips.append(bid_10_plus)

        # Age and repeat overall
        newCustomerTrips.append(int((group['customer_age_type'] == 'New').sum()))
        oldCustomerTrips.append(int((group['customer_age_type'] == 'Old').sum()))
        repeatTrips.append(int((group['repeat_type'] == 'Repeat').sum()))
        nonRepeatTrips.append(int((group['repeat_type'] == 'Non-Repeat').sum()))

        # update totals accumulator
        totals_acc['total'] += total
        totals_acc['confirmed'] += confirmed
        totals_acc['non_confirmed'] += non_confirmed
        totals_acc['completed'] += completed
        totals_acc['cancelled'] += cancelled_from_confirmed
        totals_acc['zero_bid'] += zero_bid
        # 👉 NEW
        totals_acc['bid_2_5'] += bid_2_5
        totals_acc['bid_6_10'] += bid_6_10
        totals_acc['bid_10_plus'] += bid_10_plus

        # For each car_type, age, repeat, and pickup_div combination
        for ct in car_types_with_all:
            base_ct = group if ct == 'All' else group[group['car_type'] == ct]
            for ak in age_keys:
                base_age = base_ct if ak == 'All' else base_ct[base_ct['customer_age_type'] == ak]
                for rk in repeat_keys:
                    base_repeat = base_age if rk == 'All' else base_age[base_age['repeat_type'] == rk]
                    for pdv in pickup_divs_with_all:
                        final_group = base_repeat if pdv == 'All' else base_repeat[base_repeat['pickup_div'] == pdv]

                        total_ct = int(len(final_group))
                        confirmed_ct = int(final_group['fare'].notna().sum())
                        non_confirmed_ct = int(total_ct - confirmed_ct)
                        completed_ct = int(((final_group['fare'].notna()) & (~final_group['trip_status'].isin(['Trip Confirmed', 'Cancelled']))).sum())
                        cancelled_ct = int(((final_group['fare'].notna()) & (final_group['trip_status'] == 'Cancelled')).sum())
                        zero_bid_ct = int(final_group['is_zero_bid'].sum())

                        # 👉 NEW bid counts per breakdown
                        bid_2_5_ct = int(((final_group['no_of_bids'] >= 2) & (final_group['no_of_bids'] <= 5)).sum())
                        bid_6_10_ct = int(((final_group['no_of_bids'] >= 6) & (final_group['no_of_bids'] <= 10)).sum())
                        bid_10_plus_ct = int((final_group['no_of_bids'] > 10).sum())

                        breakdown['total'][ct][ak][rk][pdv].append(total_ct)
                        breakdown['confirmed'][ct][ak][rk][pdv].append(confirmed_ct)
                        breakdown['non_confirmed'][ct][ak][rk][pdv].append(non_confirmed_ct)
                        breakdown['completed'][ct][ak][rk][pdv].append(completed_ct)
                        breakdown['cancelled_from_confirmed'][ct][ak][rk][pdv].append(cancelled_ct)
                        breakdown['zero_bid'][ct][ak][rk][pdv].append(zero_bid_ct)
                        # 👉 NEW
                        breakdown['bid_2_5'][ct][ak][rk][pdv].append(bid_2_5_ct)
                        breakdown['bid_6_10'][ct][ak][rk][pdv].append(bid_6_10_ct)
                        breakdown['bid_10_plus'][ct][ak][rk][pdv].append(bid_10_plus_ct)

    # Build daily_rows block — unchanged
    daily_rows = []
    for i, day in enumerate(grouped.groups.keys()):
        group = grouped.get_group(day)
        weekday = group['weekday'].iloc[0]
        date_label = f"{weekday}, {day.strftime('%d %b %Y')}"
        total = totalTrips[i]
        confirmed = confirmedTrips[i]
        non_confirmed = nonConfirmedTrips[i]
        completed = completedTrips[i]
        cancelled_from_confirmed = cancelledTrips[i]
        zero_bid = zeroBidTrips[i]
        trip_still_confirmed = int((group['trip_status'] == 'Trip Confirmed').sum())
        cancelled_by_customer = int((((group['fare'].notna()) & (group['trip_status'] == 'Cancelled')) & (group['cancelled_by'] == 'Customer')).sum())
        cancelled_by_driver = int((((group['fare'].notna()) & (group['trip_status'] == 'Cancelled')) & (group['cancelled_by'] == 'Driver')).sum())

        daily_rows.append({
            'date': date_label,
            'total_num': total,
            'confirmed_num': confirmed,
            'completed_num': completed,
            'cancelled_num': cancelled_from_confirmed,
            'total': total,
            'confirmed': f"{confirmed} (100%)",
            'non_confirmed': f"{non_confirmed} ({pct(non_confirmed, total)})",
            'trip_still_confirmed': f"{trip_still_confirmed} ({pct(trip_still_confirmed, confirmed)})",
            'completed': f"{completed} ({pct(completed, confirmed)})",
            'cancelled_from_confirmed': f"{cancelled_from_confirmed} ({pct(cancelled_from_confirmed, confirmed)})",
            'cancelled_by_customer': f"{cancelled_by_customer} ({pct(cancelled_by_customer, cancelled_from_confirmed)})",
            'cancelled_by_driver': f"{cancelled_by_driver} ({pct(cancelled_by_driver, cancelled_from_confirmed)})",
            'zero_bid': f"{zero_bid} ({pct(zero_bid, total)})",
        })

    totals_row = {
        'date': 'TOTAL',
        'total_num': totals_acc['total'],
        'confirmed_num': totals_acc['confirmed'],
        'completed_num': totals_acc['completed'],
        'cancelled_num': totals_acc['cancelled'],
        'total': totals_acc['total'],
        'confirmed': f"{totals_acc['confirmed']} (100%)",
        'non_confirmed': f"{totals_acc['non_confirmed']} ({pct(totals_acc['non_confirmed'], totals_acc['total'])})",
        'trip_still_confirmed': f"0 (0%)",
        'completed': f"{totals_acc['completed']} ({pct(totals_acc['completed'], totals_acc['confirmed'])})",
        'cancelled_from_confirmed': f"{totals_acc['cancelled']} ({pct(totals_acc['cancelled'], totals_acc['confirmed'])})",
        'cancelled_by_customer': f"0 (0%)",
        'cancelled_by_driver': f"0 (0%)",
        'zero_bid': f"{totals_acc['zero_bid']} ({pct(totals_acc['zero_bid'], totals_acc['total'])})"
    }
    daily_rows.append(totals_row)

    # 👉 NEW: add bid ranges to chart payload
    chart_payload = {
        'labels': [str(x) for x in dates],
        'series': {
            'total': [int(x) for x in totalTrips],
            'confirmed': [int(x) for x in confirmedTrips],
            'non_confirmed': [int(x) for x in nonConfirmedTrips],
            'completed': [int(x) for x in completedTrips],
            'cancelled_from_confirmed': [int(x) for x in cancelledTrips],
            'zero_bid': [int(x) for x in zeroBidTrips],
            'bid_2_5': [int(x) for x in bid2_5Trips],
            'bid_6_10': [int(x) for x in bid6_10Trips],
            'bid_10_plus': [int(x) for x in bid10PlusTrips],
            'new_customer': [int(x) for x in newCustomerTrips],
            'old_customer': [int(x) for x in oldCustomerTrips],
            'repeat_customer': [int(x) for x in repeatTrips],
            'non_repeat_customer': [int(x) for x in nonRepeatTrips],
        },
        'car_types': [str(x) for x in car_types_with_all],
        'pickup_divs': [str(x) for x in pickup_divs_with_all],
        'breakdown': breakdown
    }

    chart_json = json.dumps(chart_payload, default=int)




    return render(request, 'tripdata/daily_report.html', {
        'daily_rows': daily_rows,
        'chart_json': chart_json,
        'chart_car_types': car_types,
        'pickup_divs': pickup_divs
    })
