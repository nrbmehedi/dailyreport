import json
import pandas as pd
from django.shortcuts import render
from .models import TripDataUpload

def daily_report_view(request):
    trip_file = TripDataUpload.objects.filter(file_type='trip').order_by('-uploaded_at').first()
    testdata_file = TripDataUpload.objects.filter(file_type='testdata').order_by('-uploaded_at').first()

    if not trip_file or not testdata_file:
        return render(request, 'tripdata/daily_report.html', {'error': 'Please upload both Trip and TestData files first.'})

    trip_cols = ['customer_phone_no', 'fare', 'trip_status', 'cancelled_by', 'created_at', 'car_type', 'no_of_bids']
    testdata_cols = ['Testing Number']

    trip_df = pd.read_excel(trip_file.file.path, usecols=trip_cols)
    testdata_df = pd.read_excel(testdata_file.file.path, usecols=testdata_cols)

    # Clean phone numbers
    trip_df['clean_phone'] = trip_df['customer_phone_no'].astype(str).str.strip().str.replace(' ', '').str.lstrip('0')
    testdata_df['clean_test_number'] = testdata_df['Testing Number'].astype(str).str.strip().str.replace(' ', '').str.lstrip('0')
    trip_df = trip_df[~trip_df['clean_phone'].isin(testdata_df['clean_test_number'])]

    # Date and weekday
    trip_df['created_at'] = pd.to_datetime(trip_df['created_at'], errors='coerce')
    trip_df = trip_df.dropna(subset=['created_at'])
    trip_df['date'] = trip_df['created_at'].dt.date
    trip_df['weekday'] = trip_df['created_at'].dt.strftime('%A')

    # Car type cleanup
    trip_df['car_type'] = trip_df['car_type'].astype(str).fillna('Unknown').replace('nan','Unknown')
    car_types = sorted(trip_df['car_type'].unique())

    # Zero bids
    trip_df['no_of_bids'] = pd.to_numeric(trip_df['no_of_bids'], errors='coerce')
    trip_df['is_zero_bid'] = trip_df['no_of_bids'].isna() | (trip_df['no_of_bids']==0)

    # Group by date
    grouped = trip_df.groupby('date', sort=True)

    # Initialize accumulators
    daily_rows = []
    totals_acc = {k:0 for k in ['total','confirmed','non_confirmed','trip_still_confirmed','completed',
                                'cancelled','cancelled_by_customer','cancelled_by_driver','zero_bid']}

    def pct(val,parent): return f"{(val/parent*100):.0f}%" if parent else "0%"

    # Chart arrays
    dates = []
    totalTrips, confirmedTrips, completedTrips, cancelledTrips, zeroBidTrips = [], [], [], [], []
    metrics = ['total','confirmed','completed','cancelled_from_confirmed','zero_bid']
    breakdown = {metric:{ct:[] for ct in car_types} for metric in metrics}

    for day, group in grouped:
        weekday = group['weekday'].iloc[0]
        date_label = f"{weekday}, {day.strftime('%d %b %Y')}"
        dates.append(date_label)

        total = len(group)
        confirmed = group['fare'].notna().sum()
        completed = ((group['fare'].notna()) & (~group['trip_status'].isin(['Trip Confirmed','Cancelled']))).sum()
        cancelled_from_confirmed = ((group['fare'].notna()) & (group['trip_status']=='Cancelled')).sum()
        zero_bid = group['is_zero_bid'].sum()

        totalTrips.append(int(total))
        confirmedTrips.append(int(confirmed))
        completedTrips.append(int(completed))
        cancelledTrips.append(int(cancelled_from_confirmed))
        zeroBidTrips.append(int(zero_bid))

        # Totals
        totals_acc['total'] += total
        totals_acc['confirmed'] += confirmed
        totals_acc['completed'] += completed
        totals_acc['cancelled'] += cancelled_from_confirmed
        totals_acc['zero_bid'] += zero_bid

        # Car type breakdown
        for ct in car_types:
            gct = group[group['car_type']==ct]
            breakdown['total'][ct].append(len(gct))
            breakdown['confirmed'][ct].append(int(gct['fare'].notna().sum()))
            breakdown['completed'][ct].append(int(((gct['fare'].notna()) & (~gct['trip_status'].isin(['Trip Confirmed','Cancelled']))).sum()))
            breakdown['cancelled_from_confirmed'][ct].append(int(((gct['fare'].notna()) & (gct['trip_status']=='Cancelled')).sum()))
            breakdown['zero_bid'][ct].append(int(gct['is_zero_bid'].sum()))

    # Fill missing days with 0 for all car types
    nDays = len(dates)
    for ct in car_types:
        for metric in metrics:
            arr = breakdown[metric][ct]
            if len(arr) < nDays:
                arr += [0]*(nDays - len(arr))
            breakdown[metric][ct] = arr

    # Prepare table rows
    for i, day in enumerate(grouped.groups.keys()):
        group = grouped.get_group(day)
        weekday = group['weekday'].iloc[0]
        date_label = f"{weekday}, {day.strftime('%d %b %Y')}"
        total = totalTrips[i]
        confirmed = confirmedTrips[i]
        completed = completedTrips[i]
        cancelled_from_confirmed = cancelledTrips[i]
        zero_bid = zeroBidTrips[i]
        non_confirmed = total - confirmed
        trip_still_confirmed = (group['trip_status']=='Trip Confirmed').sum()
        cancelled_by_customer = (((group['fare'].notna()) & (group['trip_status']=='Cancelled')) & (group['cancelled_by']=='Customer')).sum()
        cancelled_by_driver = (((group['fare'].notna()) & (group['trip_status']=='Cancelled')) & (group['cancelled_by']=='Driver')).sum()

        daily_rows.append({
            'date': date_label,
            'total_num': total,
            'confirmed_num': confirmed,
            'completed_num': completed,
            'cancelled_num': cancelled_from_confirmed,
            'total': total,
            'confirmed': f"{confirmed} (100%)",
            'non_confirmed': f"{non_confirmed} ({pct(non_confirmed,total)})",
            'trip_still_confirmed': f"{trip_still_confirmed} ({pct(trip_still_confirmed,confirmed)})",
            'completed': f"{completed} ({pct(completed,confirmed)})",
            'cancelled_from_confirmed': f"{cancelled_from_confirmed} ({pct(cancelled_from_confirmed,confirmed)})",
            'cancelled_by_customer': f"{cancelled_by_customer} ({pct(cancelled_by_customer,cancelled_from_confirmed)})",
            'cancelled_by_driver': f"{cancelled_by_driver} ({pct(cancelled_by_driver,cancelled_from_confirmed)})",
            'zero_bid': f"{zero_bid} ({pct(zero_bid,total)})"
        })

    # TOTAL row
    totals_row = {
        'date':'TOTAL',
        'total_num': totals_acc['total'],
        'confirmed_num': totals_acc['confirmed'],
        'completed_num': totals_acc['completed'],
        'cancelled_num': totals_acc['cancelled'],
        'total': totals_acc['total'],
        'confirmed': f"{totals_acc['confirmed']} (100%)",
        'non_confirmed': f"{totals_acc['total'] - totals_acc['confirmed']} ({pct(totals_acc['total'] - totals_acc['confirmed'], totals_acc['total'])})",
        'trip_still_confirmed': f"0 (0%)",
        'completed': f"{totals_acc['completed']} ({pct(totals_acc['completed'], totals_acc['confirmed'])})",
        'cancelled_from_confirmed': f"{totals_acc['cancelled']} ({pct(totals_acc['cancelled'], totals_acc['confirmed'])})",
        'cancelled_by_customer': f"0 (0%)",
        'cancelled_by_driver': f"0 (0%)",
        'zero_bid': f"{totals_acc['zero_bid']} ({pct(totals_acc['zero_bid'], totals_acc['total'])})"
    }
    daily_rows.append(totals_row)

    chart_payload = {
        'labels': dates,
        'series': {
            'total': totalTrips,
            'confirmed': confirmedTrips,
            'completed': completedTrips,
            'cancelled_from_confirmed': cancelledTrips,
            'zero_bid': zeroBidTrips
        },
        'car_types': car_types,
        'breakdown': breakdown
    }

    return render(request, 'tripdata/daily_report.html', {
        'daily_rows': daily_rows,
        'chart_json': json.dumps(chart_payload),
        'chart_car_types': car_types  # pass car types for dropdown
    })
