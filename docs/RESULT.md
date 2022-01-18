# Result

## Compare of Argumentation Method(Hand Frame vs All Frame)
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Hyperparameter</th>
            <th>Metrics</th>
            <th>Hands Argumentation<br>(Baseline)</th>
            <th>All Frame Argumentation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>GRU-Attention</td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td>90.1</td>
            <td><b>93.1</b></td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>90.4</td>
            <td><b>93.3</b></td>
        </tr>
        <tr>
            <td rowspan=2>AdamW<br>Scheduler</td>
            <td>BLEU</td>
            <td>90.0</td>
            <td><b>93.7</b></td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>89.8</td>
            <td><b>94.2</b></td>
        </tr>
        <tr>
            <td rowspan=4>LSTM</td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td><b>73.4</b></td>
            <td>61.5</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>73.3</b></td>
            <td>61.5</td>
        </tr>
        <tr>
            <td rowspan=2>AdamW<br>Scheduler</td>
            <td>BLEU</td>
            <td><b>73.4</b></td>
            <td>52.2</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>73.3</b></td>
            <td>52.0</td>
        </tr>
    </tbody>
</table>

## Compare of Inference Method(Original vs Reverse)

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Hyperparameter</th>
            <th>Metrics</th>
            <th>Orignial<br>(Baseline)</th>
            <th>Reverse</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>GRU-Attention</td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td><b>90.1</b></td>
            <td>89.5</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>90.4</b></td>
            <td>89.7</td>
        </tr>
        <tr>
            <td rowspan=2>AdamW<br>Scheduler</td>
            <td>BLEU</td>
            <td>90.0</td>
            <td><b>90.7</b></td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>89.8</td>
            <td><b>90.6</b></td>
        </tr>
        <tr>
            <td rowspan=4>LSTM</td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td><b>73.4</b></td>
            <td>66.3</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>73.3</b></td>
            <td>66.4</td>
        </tr>
        <tr>
            <td rowspan=2>AdamW<br>Scheduler</td>
            <td>BLEU</td>
            <td><b>73.4</b></td>
            <td>64.9</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>73.3</b></td>
            <td>64.8</td>
        </tr>
    </tbody>
</table>

## Compare of Body Keypoint(Hand + Face + Body vs Hand + Body)

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Hyperparameter</th>
            <th>Metrics</th>
            <th>Hand + Face + Body<br>(Baseline)</th>
            <th>Hand + Body</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>GRU-Attention</td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td>90.1</td>
            <td><b>91.1</b></td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>90.4</td>
            <td><b>91.8</b></td>
        </tr>
        <tr>
            <td rowspan=2>AdamW<br>Scheduler</td>
            <td>BLEU</td>
            <td>90.0</td>
            <td><b>93.1</b></td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>89.8</td>
            <td><b>93.2</b></td>
        </tr>
        <tr>
            <td rowspan=4>LSTM</td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td><b>73.4</b></td>
            <td>50.7</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>73.3</b></td>
            <td>50.8</td>
        </tr>
        <tr>
            <td rowspan=2>AdamW<br>Scheduler</td>
            <td>BLEU</td>
            <td><b>73.4</b></td>
            <td>56.9</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>73.3</b></td>
            <td>56.9</td>
        </tr>
    </tbody>
</table>

## Compare of Normalization Method(Frame Normalization vs Video Normalization)

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Hyperparameter</th>
            <th>Metrics</th>
            <th>Frame Normalization<br>(Baseline)</th>
            <th>Video Normalization</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>GRU-Attention</td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td><b>90.1</b></td>
            <td>86.2</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>90.4</b></td>
            <td>86.5</td>
        </tr>
        <tr>
            <td rowspan=2>AdamW<br>Scheduler</td>
            <td>BLEU</td>
            <td><b>90.0</b></td>
            <td>89.7</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>89.8</td>
            <td><b>90.0</b></td>
        </tr>
        <tr>
            <td rowspan=4>LSTM</td>
            <td rowspan=2>Adam<br>CrossEntropy</td>
            <td>BLEU</td>
            <td><b>73.4</b></td>
            <td>59.6</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>73.3</b></td>
            <td>59.4</td>
        </tr>
        <tr>
            <td rowspan=2>AdamW<br>Scheduler</td>
            <td>BLEU</td>
            <td><b>73.4</b></td>
            <td>60.8</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td><b>73.3</b></td>
            <td>60.8</td>
        </tr>
    </tbody>
</table>

