using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using TMPro;

public class Timer : MonoBehaviour
{
    // Start is called before the first frame update
    public int minutes = 1;
    public float seconds = 0;
    public int secUI;
    public TextMeshProUGUI timerText;
    // Update is called once per frame
    void Update()
    {
        secUI = Mathf.RoundToInt(seconds);
        timerText.text = minutes.ToString() + ":" + secUI.ToString();

        seconds -= Time.deltaTime;
        print("minutes: " + minutes + "seconds: " + seconds);
        if (seconds <= 0)
        {
            if (minutes == 0)
            {
                int sceneIndex = SceneManager.GetActiveScene().buildIndex;
                SceneManager.LoadScene(sceneIndex);
            }
            else
            {
                minutes -= 1;
                seconds = 59;
            }
        }
    }
}
