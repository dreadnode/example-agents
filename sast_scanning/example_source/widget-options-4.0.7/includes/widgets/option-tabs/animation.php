<?php

/**
 * Settings Widget Options
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Add Settings Widget Options Tab
 *
 * @since 1.0
 * @return void
 */

/**
 * Called on 'extended_widget_opts_tabs'
 * create new tab navigation for alignment options
 */
function widgetopts_tab_animation($args)
{ ?>
    <li class="extended-widget-opts-tab-animation">
        <a href="#extended-widget-opts-tab-<?php echo $args['id']; ?>-animation" title="<?php _e('Animation', 'widget-options'); ?>"><span class="dashicons dashicons-admin-customizer"></span> <span class="tabtitle"><?php _e('Animation', 'widget-options'); ?></span></a>
    </li>
    <?php
}
add_action('extended_widget_opts_tabs', 'widgetopts_tab_animation', 10);

/**
 * Called on 'extended_widget_opts_tabcontent'
 * create new tab content options for alignment options
 */
if (!function_exists('widgetopts_tabcontent_animation')) :
    function widgetopts_tabcontent_animation($args)
    {
        global $widget_options;

        $id         = '';
        $classes    = '';
        $logic      = '';
        $selected   = 0;
        $check      = '';
        if (isset($args['params']) && isset($args['params']['class'])) {
            if (isset($args['params']['class']['id'])) {
                $id = $args['params']['class']['id'];
            }
            if (isset($args['params']['class']['classes'])) {
                $classes = $args['params']['class']['classes'];
            }
            if (isset($args['params']['class']['selected'])) {
                $selected = $args['params']['class']['selected'];
            }
            if (isset($args['params']['class']['logic'])) {
                $logic = $args['params']['class']['logic'];
            }
            if (isset($args['params']['class']['title']) && $args['params']['class']['title'] == '1') {
                $check = 'checked="checked"';
            }
        }

    ?>
        <div id="extended-widget-opts-tab-<?php echo $args['id']; ?>-animation" class="extended-widget-opts-tabcontent extended-widget-opts-inside-tabcontent extended-widget-opts-tabcontent-animation">

            <div class="extended-widget-opts-settings-tabs extended-widget-opts-inside-tabs">
                <input type="hidden" id="extended-widget-opts-settings-selectedtab" value="<?php echo $selected; ?>" name="<?php echo $args['namespace']; ?>[extended_widget_opts][class][selected]" />
                <!--  start tab nav -->
                <ul class="extended-widget-opts-settings-tabnav-ul" style="display: none;">


                    <li class="extended-widget-opts-settings-tab-animation">
                        <a href="#extended-widget-opts-settings-tab-<?php echo $args['id']; ?>-animation" title="<?php _e('Animation', 'widget-options'); ?>"><?php _e('Animation', 'widget-options'); ?></a>
                    </li>


                    <div class="extended-widget-opts-clearfix"></div>
                </ul><!--  end tab nav -->
                <div class="extended-widget-opts-clearfix"></div>
                <!--  start Animation tab demo -->
                <?php
                $animation_array = array(
                    'Attention Seekers' => array(
                        'bounce',
                        'flash',
                        'pulse',
                        'rubberBand',
                        'shake',
                        'swing',
                        'tada',
                        'wobble',
                        'jello'
                    ),
                    'Bouncing Entrances' => array(
                        'bounceIn',
                        'bounceInDown',
                        'bounceInLeft',
                        'bounceInRight',
                        'bounceInUp',
                    ),

                    'Fading Entrances'   => array(
                        'fadeIn',
                        'fadeInDown',
                        'fadeInDownBig',
                        'fadeInLeft',
                        'fadeInLeftBig',
                        'fadeInRight',
                        'fadeInRightBig',
                        'fadeInUp',
                        'fadeInUpBig'
                    ),
                    'Flippers'          => array(
                        'flip',
                        'flipInX',
                        'flipInY',
                        'flipOutX',
                        'flipOutY'
                    ),
                    'Lightspeed'        => array(
                        'lightSpeedIn',
                        'lightSpeedOut'
                    ),

                    'Rotating Entrances' => array(
                        'rotateIn',
                        'rotateInDownLeft',
                        'rotateInDownRight',
                        'rotateInUpLeft',
                        'rotateInUpRight'
                    ),
                    'Sliding Entrances' => array(
                        'slideInUp',
                        'slideInDown',
                        'slideInLeft',
                        'slideInRight'
                    ),
                    'Zoom Entrances'    => array(
                        'zoomIn',
                        'zoomInDown',
                        'zoomInLeft',
                        'zoomInRight',
                        'zoomInUp'
                    ),
                    'Specials'          => array(
                        'hinge',
                        'rollIn'
                    )
                ); ?>
                <!--  start animation tab content -->
                <div id="extended-widget-opts-settings-tab-<?php echo $args['id']; ?>-animation" class="extended-widget-opts-settings-tabcontent extended-widget-opts-inner-tabcontent">
                    <div class="widget-opts-animation">
                        <div class="extended-widget-opts-demo-feature">
                            <div class="extended-widget-opts-demo-warning">
                                <p class="widgetopts-unlock-features">
                                    <span class="dashicons dashicons-lock"></span><br>
                                    Unlock all Features<br>
                                    <a href="https://widget-options.com/?utm_source=wordpressadmin&amp;utm_medium=widgettabs&amp;utm_campaign=widgetoptsprotab" class="button-primary" target="_blank">Learn More</a>
                                </p>
                            </div>
                            <p>
                                <label for="opts-class-animation-<?php echo $args['id']; ?>"><?php _e('Animation Type', 'widget-options'); ?></label>
                                <br />
                                <select class="widefat" readonly>
                                    <option value=""><?php _e('None', 'widget-options'); ?></option>
                                    <?php foreach ($animation_array as $group => $anims) { ?>
                                        <optgroup label="<?php _e($group, 'widget-options'); ?>">
                                            <?php foreach ($anims as $anim => $aname) { ?>
                                                <option value="<?php echo $aname; ?>"><?php _e($aname, 'widget-options') ?></option>
                                            <?php } ?>
                                        </optgroup>
                                    <?php } ?>
                                </select>
                                <small><em><?php _e('The type of animation for this event.', 'widget-options'); ?></em></small>
                            </p>

                            <p>
                                <label for="opts-class-event-<?php echo $args['id']; ?>"><?php _e('Animation Event', 'widget-options'); ?></label>
                                <br />
                                <select class="widefat" readonly>
                                    <option value="enters"><?php _e('Element Enters Screen', 'widget-options'); ?></option>
                                    <option value="onScreen"><?php _e('Element In Screen', 'widget-options'); ?></option>
                                    <option value="pageLoad"><?php _e('Page Load', 'widget-options'); ?></option>
                                </select>
                                <small><em><?php _e('The event that triggers the animation', 'widget-options'); ?></em></small>
                            </p>

                            <p>
                                <label for="opts-class-speed-<?php echo $args['id']; ?>"><?php _e('Animation Speed', 'widget-options'); ?></label>
                                <br />
                                <input type="text" class="widefat" readonly />
                                <small><em><?php _e('How many seconds the incoming animation should lasts.', 'widget-options'); ?></em></small>
                            </p>

                            <p>
                                <label for="opts-class-offset-<?php echo $args['id']; ?>"><?php _e('Screen Offset', 'widget-options'); ?></label>
                                <br />
                                <input type="text" readonly />
                                <small><em><?php _e('How many pixels above the bottom of the screen must the widget be before animating.', 'widget-options'); ?></em></small>
                            </p>

                            <p>
                                <label for="opts-class-hidden-<?php echo $args['id']; ?>"><?php _e('Hide Before Animation', 'widget-options'); ?></label>
                                <br />
                                <input type="checkbox" value="1" readonly />
                                <label for="opts-class-hidden-<?php echo $args['id']; ?>"><?php _e('Enabled', 'widget-options'); ?></label><br />
                                <small><em><?php _e('Hide widget before animating.', 'widget-options'); ?></em></small>
                            </p>

                            <p>
                                <label for="opts-class-delay-<?php echo $args['id']; ?>"><?php _e('Animation Delay', 'widget-options'); ?></label>
                                <br />
                                <input type="text" class="widefat" readonly />
                                <small><em><?php _e('Number of seconds after the event to start the animation.', 'widget-options'); ?></em></small>
                            </p>
                        </div>
                    </div>
                </div><!--  end animation tab content -->
                <!--  end Animation tab demo -->

            </div><!-- end .extended-widget-opts-settings-tabs -->


        </div>
<?php
    }
    add_action('extended_widget_opts_tabcontent', 'widgetopts_tabcontent_animation');
endif; ?>